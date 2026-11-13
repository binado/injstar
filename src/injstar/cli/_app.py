"""Command-line interface.

``injstar snr CONFIG`` simulates every injection of a population and writes
their per-detector and network SNRs and detection flags. ``injstar run CONFIG``
also localizes the detected ones with BAYESTAR. ``injstar run CONFIG --resume``
finishes an interrupted run and ``injstar summarise DIR`` rebuilds its summary
table::

    injstar snr run.toml --output-dir out
    injstar run run.toml --output-dir out
    injstar run run.toml --output-dir out --resume
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
ResumeOption = Annotated[
    bool,
    typer.Option(
        "--resume", help="Continue an interrupted run, skipping finished events."
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
