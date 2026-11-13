"""The ``snr``, ``run``, ``resume`` and ``summarise`` commands."""

import typer

from injstar.cli._app import (
    ConfigArgument,
    OutputDirOption,
    OverwriteOption,
    RunDirectoryArgument,
    SeedOption,
    WorkersOption,
    _guarded,
    app,
)


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
