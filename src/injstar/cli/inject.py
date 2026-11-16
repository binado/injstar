"""``injstar inject``: Compute SNRs and detection flags."""

import typer

from injstar.cli._app import (
    ConfigArgument,
    OutputDirOption,
    OverwriteOption,
    SeedOption,
    _guarded,
    app,
)


@app.command()
def inject(
    config: ConfigArgument,
    output_dir: OutputDirOption = None,
    seed: SeedOption = None,
    overwrite: OverwriteOption = False,
) -> None:
    """Compute the SNRs and detection flags of every injection."""

    def stage() -> None:
        from gwmock_pop.loaders.file_loader import read_population_catalogue

        from injstar.config import load_inject_config

        loaded = load_inject_config(config, output_dir=output_dir, seed=seed)
        path = loaded.run(overwrite=overwrite)
        columns = read_population_catalogue(path)
        detected = int(columns["detected"].sum())
        typer.echo(f"{detected} of {len(columns['detected'])} injections detected")
        typer.echo(path)

    _guarded(stage)
