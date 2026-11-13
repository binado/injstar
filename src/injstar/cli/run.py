"""``injstar run``: Simulate and localize a population."""

import typer

from injstar.cli._app import (
    ConfigArgument,
    OutputDirOption,
    OverwriteOption,
    ResumeOption,
    SeedOption,
    WorkersOption,
    _guarded,
    app,
)


@app.command()
def run(
    config: ConfigArgument,
    output_dir: OutputDirOption = None,
    workers: WorkersOption = None,
    seed: SeedOption = None,
    overwrite: OverwriteOption = False,
    resume: ResumeOption = False,
) -> None:
    """Simulate and localize a population of injections."""

    def stage() -> None:
        from injstar import pipeline
        from injstar.config import load_run_config

        loaded = load_run_config(
            config, output_dir=output_dir, seed=seed, workers=workers
        )
        typer.echo(pipeline.run(loaded, overwrite=overwrite, resume=resume))

    _guarded(stage)
