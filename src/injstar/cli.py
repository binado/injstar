"""Command-line interface.

``injstar snr CONFIG`` simulates every injection of a population and writes
their per-detector and network SNRs and detection flags. ``injstar run CONFIG``
also localizes the detected ones with BAYESTAR. ``injstar resume DIR`` finishes
an interrupted run and ``injstar summarise DIR`` rebuilds its summary table::

    injstar snr run.toml --output-dir out
    injstar run run.toml --output-dir out
    injstar resume out
    injstar summarise out

This module must stay cheap to import: JAX, ligo.skymap and the pipeline are
imported inside the commands, never here.
"""

import logging
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Annotated

import typer

from injstar import __version__

app = typer.Typer(
    help=(
        "Simulated sky-distance posteriors with BAYESTAR for arbitrary "
        "detector networks."
    ),
    no_args_is_help=True,
)

ConfigArgument = Annotated[
    Path,
    typer.Argument(exists=True, dir_okay=False, help="Run config TOML file."),
]
RunDirectoryArgument = Annotated[
    Path,
    typer.Argument(exists=True, file_okay=False, help="Output directory of the run."),
]
OutputDirOption = Annotated[
    Path | None,
    typer.Option("--output-dir", help="Output directory, replacing the config's."),
]
SeedOption = Annotated[
    int | None, typer.Option("--seed", min=0, help="Root seed, replacing the config's.")
]
OverwriteOption = Annotated[
    bool,
    typer.Option(
        "--overwrite", help="Replace an existing run in the output directory."
    ),
]
WorkersOption = Annotated[
    int | None,
    typer.Option("--workers", min=1, help="BAYESTAR worker processes."),
]


def _version(value: bool) -> None:
    if value:
        typer.echo(f"injstar {__version__}")
        raise typer.Exit


@app.callback()
def _main(
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            callback=_version,
            is_eager=True,
            help="Show the version and exit.",
        ),
    ] = False,
    verbose: Annotated[
        bool, typer.Option("--verbose", "-v", help="Log progress to stderr.")
    ] = False,
) -> None:
    """Simulated sky-distance posteriors with BAYESTAR."""
    logging.basicConfig(format="%(asctime)s %(name)s: %(message)s")
    logging.getLogger("injstar").setLevel(logging.INFO if verbose else logging.WARNING)


def _guarded[T](stage: Callable[[], T]) -> T:
    """Run a stage, turning input and run-directory errors into CLI errors."""
    # The pipeline pulls in numpy, h5py and pydantic; keep --help instant.
    from pydantic import ValidationError

    from injstar.pipeline import RunDirectoryError

    try:
        return stage()
    except (ValidationError, RunDirectoryError, ValueError, FileNotFoundError) as error:
        typer.secho(f"Error: {error}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from error


@app.command()
def snr(
    config: ConfigArgument,
    output_dir: OutputDirOption = None,
    seed: SeedOption = None,
    overwrite: OverwriteOption = False,
) -> None:
    """Compute the SNRs and detection flags of every injection."""

    def stage() -> None:
        from injstar import pipeline, products
        from injstar.config import load_run_config

        loaded = load_run_config(config, output_dir=output_dir, seed=seed)
        path = pipeline.compute_snr(loaded, overwrite=overwrite)
        columns, _ = products.read_table(path)
        detected = int(columns["detected"].sum())
        typer.echo(f"{detected} of {len(columns['detected'])} injections detected")
        typer.echo(path)

    _guarded(stage)


@app.command()
def run(
    config: ConfigArgument,
    output_dir: OutputDirOption = None,
    workers: WorkersOption = None,
    seed: SeedOption = None,
    overwrite: OverwriteOption = False,
) -> None:
    """Simulate and localize a population of injections."""

    def stage() -> None:
        from injstar import pipeline
        from injstar.config import load_run_config

        loaded = load_run_config(
            config, output_dir=output_dir, seed=seed, workers=workers
        )
        typer.echo(pipeline.run(loaded, overwrite=overwrite))

    _guarded(stage)


@app.command()
def resume(output_dir: RunDirectoryArgument, workers: WorkersOption = None) -> None:
    """Continue an interrupted run, skipping finished events."""

    def stage() -> None:
        from injstar import pipeline

        typer.echo(pipeline.resume(output_dir, workers=workers))

    _guarded(stage)


@app.command()
def summarise(output_dir: RunDirectoryArgument) -> None:
    """Rebuild the summary table from a run's sky maps."""

    def stage() -> None:
        from injstar import pipeline

        typer.echo(pipeline.summarise(output_dir))

    _guarded(stage)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the ``injstar`` command and return its exit code.

    Parameters
    ----------
    argv
        Arguments after the program name; ``None`` reads ``sys.argv``.
    """
    try:
        app(args=None if argv is None else list(argv), standalone_mode=False)
    except typer.Exit as error:
        return error.exit_code
    except typer.Abort:
        return 1
    return 0
