"""Combine explicitly supplied localization summaries."""

from pathlib import Path
from typing import Annotated

import typer

from injstar.cli._app import OverwriteOption, _guarded, app


@app.command()
def collect_summary(
    directories: Annotated[
        list[Path], typer.Argument(help="Localization directories.")
    ],
    output: Annotated[Path, typer.Option(help="Destination summary CSV.")],
    overwrite: OverwriteOption = False,
) -> None:
    """Combine chunk summaries, sorted by original population row index."""
    from injstar.products import collect_summaries

    typer.echo(
        _guarded(lambda: collect_summaries(directories, output, overwrite=overwrite))
    )
