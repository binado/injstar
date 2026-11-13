"""``injstar resume``: Continue an interrupted run."""

import typer

from injstar.cli._app import (
    RunDirectoryArgument,
    WorkersOption,
    _guarded,
    app,
)


@app.command()
def resume(output_dir: RunDirectoryArgument, workers: WorkersOption = None) -> None:
    """Continue an interrupted run, skipping finished events."""

    def stage() -> None:
        from injstar import pipeline

        typer.echo(pipeline.resume(output_dir, workers=workers))

    _guarded(stage)
