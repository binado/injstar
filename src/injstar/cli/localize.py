"""``injstar localize``: Localize saved injections."""

import typer

from injstar.cli._app import (
    ConfigArgument,
    OutputDirOption,
    OverwriteOption,
    ResumeOption,
    WorkersOption,
    _guarded,
    app,
)


@app.command()
def localize(
    config: ConfigArgument,
    output_dir: OutputDirOption = None,
    workers: WorkersOption = None,
    overwrite: OverwriteOption = False,
    resume: ResumeOption = False,
) -> None:
    """Localize saved injections with BAYESTAR and write summary.csv."""

    def stage() -> None:
        from injstar.config import load_localize_config

        loaded = load_localize_config(config, output_dir=output_dir, workers=workers)
        typer.echo(loaded.run(overwrite=overwrite, resume=resume))

    _guarded(stage)
