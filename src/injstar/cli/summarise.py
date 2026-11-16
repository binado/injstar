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
        from injstar import products
        from injstar.config._common import RunDirectoryError

        if not (output_dir / "injections.h5").is_file():
            raise RunDirectoryError(f"{output_dir} has no injections.h5")

        typer.echo(products.build_summary(output_dir))

    _guarded(stage)
