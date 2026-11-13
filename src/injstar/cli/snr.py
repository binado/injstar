"""``injstar snr``: Compute SNRs and detection flags."""

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
