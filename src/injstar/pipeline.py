"""The stages behind the ``injstar`` commands.

:func:`compute_snr` simulates every injection and writes ``injections.h5``.
:func:`run` adds one BAYESTAR sky map per detected event and the summary.
Simulation is deterministic per event, so :func:`run` re-draws the SNR series
of the events it localizes instead of storing them, and an interrupted run is
finished by calling it again with the same config and ``resume=True``.

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
from collections.abc import Mapping
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np

from injstar import products
from injstar.config import RunConfig
from injstar.detector.network import DetectorRegistry, load_detector_config
from injstar.population import Population, read_population, simulation_events
from injstar.runtime import configure_runtime, set_cpu_threads

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
    population = read_population(
        config.population, cosmology=config.population_cosmology
    )
    return Inputs(config, registry, population)


def _results_key(config: Mapping[str, Any]) -> dict[str, Any]:
    """The part of a dumped config that defines a run's results."""
    return {k: v for k, v in config.items() if k not in _EXECUTION_FIELDS}


def _prepare_directory(config: RunConfig, *, overwrite: bool) -> bool:
    """Make the output directory usable; return whether it holds this run."""
    output_dir = config.output_dir
    snapshot = output_dir / products.CONFIG_FILE
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
    (config.output_dir / products.CONFIG_FILE).write_text(
        config.model_dump_json(indent=2) + "\n"
    )


def _simulator(inputs: Inputs):  # noqa: ANN202 - simulate imports JAX
    """Build the simulator; JAX is configured and imported here, not earlier."""
    configure_runtime()
    from injstar.simulate import Simulator, greenwich_sidereal_time

    events = simulation_events(
        inputs.population, greenwich_sidereal_time(inputs.population["coa_time"])
    )
    resolved_config = inputs.config.model_copy(
        update={"network": inputs.registry.members("selected")}
    )
    return Simulator.from_config(resolved_config, inputs.registry), events


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
    target = config.output_dir / products.INJECTIONS_FILE
    if _prepare_directory(config, overwrite=overwrite) and target.is_file():
        logger.info("reusing %s", target)
        return target

    simulator, events = _simulator(inputs)
    result = simulator.snr(events)
    columns = dict(inputs.population)
    for index, name in enumerate(simulator.names):
        columns[f"expected_snr_{name}"] = result.expected_snr[:, index]
        columns[f"observed_snr_{name}"] = result.observed_snr[:, index]
        columns[f"horizon_{name}"] = result.horizon[:, index]
    columns["network_snr"] = result.network_snr
    columns["detected"] = result.detected

    _write_config(config)
    products.write_table(target, columns)
    logger.info(
        "%d of %d injections detected", int(result.detected.sum()), len(result.detected)
    )
    return target


def _pending(output_dir: Path, event_ids: np.ndarray) -> np.ndarray:
    """Indices of the events that have no sky map yet."""
    return np.array(
        [
            index
            for index, event_id in enumerate(event_ids)
            if not products.skymap_path(output_dir, int(event_id)).is_file()
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
    columns, _ = products.read_table(output_dir / products.INJECTIONS_FILE)
    detected = np.flatnonzero(columns["detected"])
    todo = detected[_pending(output_dir, columns["event_id"][detected])]
    logger.info(
        "%d detected, %d already localized, %d to do",
        len(detected),
        len(detected) - len(todo),
        len(todo),
    )

    if len(todo):
        # ligo.skymap is imported here so that `injstar snr` never loads it.
        from injstar.localize import Event, Network, localize_to_file

        simulator, events = _simulator(inputs)
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
                drawn = simulator.series(
                    {name: values[rows] for name, values in events.items()}
                )
                batch = [
                    Event(
                        event_id=int(columns["event_id"][row]),
                        coa_time=float(columns["coa_time"][row]),
                        series=drawn.series[index],
                        epochs=drawn.epochs[index],
                        snr=drawn.scalars.observed_snr[index],
                        bandwidth=drawn.scalars.bandwidth[index],
                        horizon=drawn.scalars.horizon[index],
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
    if not (output_dir / products.INJECTIONS_FILE).is_file():
        raise RunDirectoryError(f"{output_dir} has no {products.INJECTIONS_FILE}")
    return products.build_summary(output_dir)
