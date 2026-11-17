"""Stage configuration and file workflows."""

from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Self

from pydantic import BaseModel, Field, model_validator

if TYPE_CHECKING:
    import numpy as np

from ._common import _STRICT, Positive, load_config, prepare_directory, write_config


class PriorConfig(BaseModel):
    """BAYESTAR distance prior. Distances are in Mpc."""

    model_config = _STRICT

    min_distance: Annotated[float, Field(ge=0)] = 0.0
    max_distance: Positive
    distance_power: int = 2
    cosmology: bool = False

    @model_validator(mode="after")
    def _validate(self) -> Self:
        if self.max_distance <= self.min_distance:
            raise ValueError("max_distance must exceed min_distance")
        if self.cosmology and self.distance_power != 2:
            raise ValueError("cosmology = true requires distance_power = 2")
        if self.min_distance == 0 and self.distance_power < 0:
            raise ValueError("a prior r^k with k < 0 needs min_distance > 0")
        return self


class LocalizeConfig(BaseModel):
    """Localization workflow consuming completed injection products."""

    model_config = _STRICT

    injection_dir: Path
    network: str | tuple[str, ...]
    detector_config: tuple[Path, ...] = ()
    output_dir: Path
    prior: PriorConfig
    rescale_loglikelihood: Positive = 1.0
    batch_size: Annotated[int, Field(ge=1)] = 32
    workers: Annotated[int, Field(ge=1)] = 1
    threads_per_worker: Annotated[int, Field(ge=1)] = 1

    @model_validator(mode="after")
    def _validate(self) -> Self:
        source, target = self.injection_dir.resolve(), self.output_dir.resolve()
        if source == target or source in target.parents or target in source.parents:
            raise ValueError("injection and localization directories must not overlap")
        return self

    def run(self, *, overwrite: bool = False, resume: bool = False) -> Path:
        """Localize saved detections and return the summary catalogue."""
        return _run(self, overwrite=overwrite, resume=resume)


def load_localize_config(path: Path, **overrides: Any) -> LocalizeConfig:
    """Load localization TOML, resolving paths relative to the file."""
    return load_config(path, LocalizeConfig, **overrides)


def _pending(output_dir: Path, rows: "np.ndarray") -> "np.ndarray":
    """Indices of the events that have no sky map yet."""
    import numpy as np

    from injstar import io

    return np.array(
        [
            index
            for index, row in enumerate(rows)
            if not io.skymap_path(output_dir, int(row)).is_file()
        ],
        dtype=int,
    )


def _run(
    config: LocalizeConfig, *, overwrite: bool = False, resume: bool = False
) -> Path:
    """Localize the detected injections and write the summary.

    Parameters
    ----------
    config
        Run configuration.
    overwrite
        Replace an output directory holding a different or finished run.
    resume
        Continue a directory that already holds sky maps of this run. The
        directory's recorded config must match ``config``.

    Returns
    -------
    pathlib.Path
        The ``summary.csv`` of the run directory.

    Raises
    ------
    RunDirectoryError
        If the output directory holds something else, or sky maps of this run
        when neither ``overwrite`` nor ``resume`` is set.
    ValueError
        If both ``overwrite`` and ``resume`` are set.
    """
    import logging
    import multiprocessing
    import os
    from concurrent.futures import ProcessPoolExecutor
    from functools import partial

    import h5py
    import numpy as np
    from gwmock_pop.loaders.file_loader import read_population_catalogue

    from injstar import io, products
    from injstar.detector.config import DetectorRegistry
    from injstar.runtime import set_cpu_threads

    from ._common import RunDirectoryError

    logger = logging.getLogger(__name__)
    if overwrite and resume:
        raise ValueError("overwrite and resume are mutually exclusive")
    output_dir = config.output_dir
    skymaps = output_dir / "skymaps"
    source = config.injection_dir
    for name in (io.CONFIG_FILE, io.INJECTIONS_FILE, io.INJECTION_RESULTS_FILE):
        if not (source / name).is_file():
            raise RunDirectoryError(f"{source} has no completed {name}")
    registry = DetectorRegistry.load(config.detector_config).select(config.network)
    with h5py.File(source / io.INJECTION_RESULTS_FILE, "r") as saved:
        names = tuple(saved.attrs["detectors"])
        rate = float(saved.attrs["sampling_frequency"])
        detected_positions = np.flatnonzero(saved["detected"][:])
        saved_rows = saved["row_index"][:]
    if tuple(registry.detectors) != names:
        raise ValueError("configured detector order does not match saved detector axis")
    prepare_directory(config, overwrite=overwrite)
    if skymaps.is_dir() and any(skymaps.iterdir()) and not resume:
        raise RunDirectoryError(
            f"{output_dir} already holds sky maps of this run; "
            "pass --resume to finish it or --overwrite to redo it"
        )

    write_config(config)
    catalogue = output_dir / io.INJECTIONS_FILE
    if not catalogue.is_symlink():
        catalogue.symlink_to(os.path.relpath(source / io.INJECTIONS_FILE, output_dir))
    columns = read_population_catalogue(catalogue)
    detected = saved_rows[detected_positions]
    todo_positions = detected_positions[_pending(output_dir, detected)]
    todo = saved_rows[todo_positions]
    logger.info(
        "%d detected, %d already localized, %d to do",
        len(detected),
        len(detected) - len(todo),
        len(todo),
    )

    if len(todo):
        from injstar.dataset import InjectionResult
        from injstar.localize import Event, Network, localize_to_file

        names, responses, locations = registry.arrays("selected")
        network = Network(
            names=names,
            responses=responses,
            locations=locations,
            sampling_frequency=rate,
        )
        task = partial(
            localize_to_file,
            network=network,
            output_dir=output_dir,
            **config.prior.model_dump(),
            rescale_loglikelihood=config.rescale_loglikelihood,
        )
        with ProcessPoolExecutor(
            max_workers=config.workers,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=set_cpu_threads,
            initargs=(config.threads_per_worker,),
        ) as pool:
            for start in range(0, len(todo), config.batch_size):
                rows = todo[start : start + config.batch_size]
                positions = todo_positions[start : start + config.batch_size]
                drawn = InjectionResult.read(
                    source / io.INJECTION_RESULTS_FILE,
                    selection=positions,
                )
                batch = [
                    Event(
                        row_index=int(row),
                        right_ascension=float(columns["right_ascension"][position]),
                        declination=float(columns["declination"][position]),
                        distance=float(columns["distance"][position]),
                        coa_time=float(columns["coa_time"][position]),
                        series=np.asarray(drawn.series[index]),
                        epochs=np.asarray(drawn.epochs[index]),
                        snr=np.asarray(drawn.observed_snr[index]),
                        bandwidth=np.asarray(drawn.bandwidth[index]),
                        horizon=np.asarray(drawn.horizon[index]),
                    )
                    for index, (row, position) in enumerate(
                        zip(rows, positions, strict=True)
                    )
                ]
                for _ in pool.map(task, batch):
                    pass
                logger.info(
                    "localized %d of %d", min(start + len(rows), len(todo)), len(todo)
                )

    return products.build_summary(output_dir)
