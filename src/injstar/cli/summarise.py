"""``injstar summarise``: Rebuild a run's summary table."""

import typer

from injstar.cli._app import (
    RunDirectoryArgument,
    _guarded,
    app,
)


@app.command()
def summarise(output_dir: RunDirectoryArgument) -> None:
    """Rebuild the summary table from a run's sky maps."""

    def stage() -> None:
        from injstar import pipeline

        typer.echo(pipeline.summarise(output_dir))

    _guarded(stage)
