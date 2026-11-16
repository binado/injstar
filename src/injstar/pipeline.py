"""The stages behind the ``injstar`` commands.

:func:`compute_snr` simulates every injection and writes ``injections.h5``.
:func:`run` adds one BAYESTAR sky map per detected event and the summary.
Complete injection results are stored in HDF5 and read in batches for
localization. An interrupted run is finished by calling it again with the same
config and ``resume=True``.

A run directory holds ``config.json``, the :class:`~injstar.config.RunConfig`
that produced it. A directory is only reused when that config matches the
requested one, ignoring the fields that change how a run executes but not what
it produces. The population and PSD files are not hashed, so editing them in
place is not detected.
"""

import json
import logging
import multiprocessing
import shutil
from collections.abc import Iterator, Mapping
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from gwmock_pop.loaders.file_loader import read_population_catalogue

from injstar import io, products
from injstar.config import RunConfig
from injstar.detector.network import DetectorRegistry, load_detector_config
from injstar.population import Population, read_population, simulation_events
from injstar.runtime import configure_runtime, set_cpu_threads

if TYPE_CHECKING:
    from injstar.dataset import InjectionResult

logger = logging.getLogger(__name__)

#: Config fields that change how a run executes but not what it produces.
_EXECUTION_FIELDS = frozenset(
    {"output_dir", "workers", "threads_per_worker", "batch_size"}
)


class RunDirectoryError(RuntimeError):
    """The output directory cannot be used for the requested stage."""


@dataclass(frozen=True)
class Inputs:
    """Validated inputs of a run.

    Attributes
    ----------
    config
        Run configuration.
    registry
        Registry restricted to the configured network.
    population
        Canonical population columns.
    """

    config: RunConfig
    registry: DetectorRegistry
    population: Population


def load_inputs(config: RunConfig) -> Inputs:
    """Read the detector registry and population a config points to.

    Parameters
    ----------
    config
        Run configuration.
    """
    registry = load_detector_config(config.detector_config).select(config.network)
    population = read_population(config.population)
    generated = {
        "network_snr",
        "detected",
        "localized",
        *products.statistic_names(),
    }
    for name in registry.members("selected"):
        generated.update(
            f"{prefix}_{name}" for prefix in ("expected_snr", "observed_snr", "horizon")
        )
    collisions = generated.intersection(population)
    if collisions:
        raise ValueError(
            f"population columns collide with generated outputs: {sorted(collisions)}"
        )
    return Inputs(config, registry, population)


def _results_key(config: Mapping[str, Any]) -> dict[str, Any]:
    """The part of a dumped config that defines a run's results."""
    return {k: v for k, v in config.items() if k not in _EXECUTION_FIELDS}


def _prepare_directory(config: RunConfig, *, overwrite: bool) -> bool:
    """Make the output directory usable; return whether it holds this run."""
    output_dir = config.output_dir
    snapshot = output_dir / io.CONFIG_FILE
    occupied = output_dir.exists() and any(output_dir.iterdir())
    recorded = json.loads(snapshot.read_text()) if snapshot.is_file() else None
    requested = config.model_dump(mode="json")
    if recorded is not None and _results_key(recorded) == _results_key(requested):
        if not overwrite:
            return True
    elif occupied and not overwrite:
        reason = (
            "holds a run with a different config"
            if recorded is not None
            else "is not empty and is not an injstar run"
        )
        raise RunDirectoryError(
            f"{output_dir} {reason}; pass --overwrite to replace it"
        )
    if occupied:
        if recorded is None:
            raise RunDirectoryError(
                f"{output_dir} is not an injstar run directory; refusing to delete it"
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return False


def _write_config(config: RunConfig) -> None:
    """Record the config in the run directory."""
    (config.output_dir / io.CONFIG_FILE).write_text(
        config.model_dump_json(indent=2) + "\n"
    )


def _simulation_batches(inputs: Inputs) -> Iterator["InjectionResult"]:
    """Configure JAX and yield results from one prepared simulation kernel."""
    configure_runtime()
    from injstar.simulate import _inject_batches, greenwich_sidereal_time
    from injstar.waveform import waveform

    config = inputs.config.model_copy(
        update={"network": inputs.registry.members("selected")}
    )
    events = simulation_events(
        inputs.population, greenwich_sidereal_time(inputs.population["coa_time"])
    )
    return _inject_batches(
        events,
        waveform=waveform(
            config.waveform_approximant, reference_frequency=config.reference_frequency
        ),
        template=waveform(
            config.template_approximant or config.waveform_approximant,
            reference_frequency=config.reference_frequency,
        ),
        network=inputs.registry,
        config=config,
    )


def compute_snr(config: RunConfig, *, overwrite: bool = False) -> Path:
    """Simulate every injection and write ``injections.h5``.

    Parameters
    ----------
    config
        Run configuration.
    overwrite
        Replace an output directory holding a different or finished run.

    Returns
    -------
    pathlib.Path
        The ``injections.h5`` of the run directory, reused if it already
        holds this run.

    Raises
    ------
    RunDirectoryError
        If the output directory holds something else and ``overwrite`` is
        false.
    """
    inputs = load_inputs(config)
    target = config.output_dir / io.INJECTIONS_FILE
    reused = _prepare_directory(config, overwrite=overwrite)
    results_path = config.output_dir / io.INJECTION_RESULTS_FILE
    if reused and target.is_file() and results_path.is_file():
        logger.info("reusing %s", target)
        return target

    _write_config(config)
    partial_results = results_path.with_suffix(".partial.h5")
    partial_catalogue = target.with_suffix(".partial.h5")
    results_path.unlink(missing_ok=True)
    partial_results.unlink(missing_ok=True)
    chunks: dict[str, list[np.ndarray]] = {}
    try:
        for result in _simulation_batches(inputs):
            result.write(partial_results, append=True)
            batch_columns = {}
            for index, name in enumerate(result.detectors):
                for field in ("expected_snr", "observed_snr", "horizon"):
                    batch_columns[f"{field}_{name}"] = np.asarray(
                        getattr(result, field)
                    )[:, index]
            batch_columns["network_snr"] = np.asarray(result.network_snr)
            batch_columns["detected"] = np.asarray(result.detected)
            for name, values in batch_columns.items():
                chunks.setdefault(name, []).append(values)
        columns = dict(inputs.population)
        columns.update({name: np.concatenate(parts) for name, parts in chunks.items()})
        io.write_table(partial_catalogue, columns)
        partial_catalogue.replace(target)
        partial_results.replace(results_path)
    finally:
        partial_results.unlink(missing_ok=True)
        partial_catalogue.unlink(missing_ok=True)
    logger.info(
        "%d of %d injections detected",
        int(columns["detected"].sum()),
        len(columns["detected"]),
    )
    return target


def _pending(output_dir: Path, rows: np.ndarray) -> np.ndarray:
    """Indices of the events that have no sky map yet."""
    return np.array(
        [
            index
            for index, row in enumerate(rows)
            if not io.skymap_path(output_dir, int(row)).is_file()
        ],
        dtype=int,
    )


def run(config: RunConfig, *, overwrite: bool = False, resume: bool = False) -> Path:
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
        The ``summary.h5`` of the run directory.

    Raises
    ------
    RunDirectoryError
        If the output directory holds something else, or sky maps of this run
        when neither ``overwrite`` nor ``resume`` is set.
    ValueError
        If both ``overwrite`` and ``resume`` are set.
    """
    if overwrite and resume:
        raise ValueError("overwrite and resume are mutually exclusive")
    output_dir = config.output_dir
    skymaps = output_dir / "skymaps"
    compute_snr(config, overwrite=overwrite)
    if skymaps.is_dir() and any(skymaps.iterdir()) and not resume:
        raise RunDirectoryError(
            f"{output_dir} already holds sky maps of this run; "
            "pass --resume to finish it or --overwrite to redo it"
        )

    inputs = load_inputs(config)
    columns = read_population_catalogue(output_dir / io.INJECTIONS_FILE)
    detected = np.flatnonzero(columns["detected"])
    todo = detected[_pending(output_dir, detected)]
    logger.info(
        "%d detected, %d already localized, %d to do",
        len(detected),
        len(detected) - len(todo),
        len(todo),
    )

    if len(todo):
        # ligo.skymap is imported here so that `injstar snr` never loads it.
        from injstar.dataset import InjectionResult
        from injstar.localize import Event, Network, localize_to_file

        names, responses, locations = inputs.registry.arrays("selected")
        network = Network(
            names=names,
            responses=responses,
            locations=locations,
            sampling_frequency=config.sampling_frequency,
            prior=config.prior,
            rescale_loglikelihood=config.rescale_loglikelihood,
        )
        task = partial(localize_to_file, network=network, output_dir=output_dir)
        with ProcessPoolExecutor(
            max_workers=config.workers,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=set_cpu_threads,
            initargs=(config.threads_per_worker,),
        ) as pool:
            for start in range(0, len(todo), config.batch_size):
                rows = todo[start : start + config.batch_size]
                drawn = InjectionResult.read(
                    output_dir / io.INJECTION_RESULTS_FILE, selection=rows
                )
                batch = [
                    Event(
                        row_index=int(row),
                        coa_time=float(columns["coa_time"][row]),
                        series=np.asarray(drawn.series[index]),
                        epochs=np.asarray(drawn.epochs[index]),
                        snr=np.asarray(drawn.observed_snr[index]),
                        bandwidth=np.asarray(drawn.bandwidth[index]),
                        horizon=np.asarray(drawn.horizon[index]),
                    )
                    for index, row in enumerate(rows)
                ]
                for _ in pool.map(task, batch):
                    pass
                logger.info(
                    "localized %d of %d", min(start + len(rows), len(todo)), len(todo)
                )

    return products.build_summary(output_dir)


def summarise(output_dir: Path) -> Path:
    """Rebuild ``summary.h5`` from ``injections.h5`` and the sky maps.

    Parameters
    ----------
    output_dir
        Run directory.

    Raises
    ------
    RunDirectoryError
        If the directory has no ``injections.h5``.
    """
    output_dir = Path(output_dir)
    if not (output_dir / io.INJECTIONS_FILE).is_file():
        raise RunDirectoryError(f"{output_dir} has no {io.INJECTIONS_FILE}")
    return products.build_summary(output_dir)
