"""Injection configuration and streaming file workflow."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Self

from pydantic import BaseModel, Field, model_validator

from ._common import _STRICT, Positive, load_config, prepare_directory, write_config

if TYPE_CHECKING:
    from injstar.dataset import InjectionResult
    from injstar.detector.network import DetectorRegistry
    from injstar.population import Population

logger = logging.getLogger(__name__)


class DetectionConfig(BaseModel):
    """Detection rule applied to the observed SNRs.

    An event is detected when at least ``min_detectors`` detectors reach
    ``detector_snr`` and the network SNR reaches ``network_snr``.
    """

    model_config = _STRICT

    detector_snr: Annotated[float, Field(ge=0)] = 4.0
    network_snr: Annotated[float, Field(ge=0)] = 8.0
    min_detectors: Annotated[int, Field(ge=1)] = 2


class InjectConfig(BaseModel):
    """Everything that defines a run.

    Attributes
    ----------
    network
        A network name from the detector registry, or a list of detector names.
    detector_config
        Detector registry TOML layers merged over the packaged defaults.
    population
        Population file (CSV or HDF5).
    output_dir
        Run directory.
    waveform_approximant
        Waveform model of the injected signal; any ripple frequency-domain
        compact-binary model, including precessing and higher-mode ones.
    template_approximant
        Waveform model of the matched-filter template, which must be a
        dominant-mode aligned-spin model (BAYESTAR's signal model is the
        dominant mode). Defaults to ``waveform_approximant``, which then has to
        be one itself.
    reference_frequency
        Reference frequency of the spins and phase in Hz.
    minimum_frequency, maximum_frequency
        Frequency band of the matched filter in Hz.
    sampling_frequency
        Sample rate of the SNR time series in Hz.
    segment_duration
        Length in seconds of the periodic segment the SNR series is drawn on;
        the frequency resolution is its inverse. Must be long enough for the
        template autocorrelation to decay.
    crop_duration
        Length in seconds of the SNR series handed to BAYESTAR.
    noise
        Whether to add Gaussian noise to the SNR series.
    """

    model_config = _STRICT

    network: str | tuple[str, ...]
    detector_config: tuple[Path, ...] = ()
    population: Path
    output_dir: Path

    waveform_approximant: str = "IMRPhenomD"
    template_approximant: str | None = None
    reference_frequency: Positive = 20.0
    minimum_frequency: Positive = 20.0
    maximum_frequency: Positive = 1024.0
    sampling_frequency: Positive = 4096.0
    segment_duration: Positive = 8.0
    crop_duration: Positive = 0.1
    noise: bool = True

    detection: DetectionConfig = DetectionConfig()

    seed: Annotated[int, Field(ge=0)] = 0
    batch_size: Annotated[int, Field(ge=1)] = 32

    @model_validator(mode="after")
    def _validate(self) -> Self:
        if self.maximum_frequency <= self.minimum_frequency:
            raise ValueError("maximum_frequency must exceed minimum_frequency")
        if self.maximum_frequency > self.sampling_frequency / 2:
            raise ValueError("maximum_frequency must not exceed sampling_frequency / 2")
        samples = self.sampling_frequency * self.segment_duration
        if abs(samples - round(samples)) > 1e-9 or round(samples) % 2:
            raise ValueError(
                "sampling_frequency * segment_duration must be an even whole number of samples"
            )
        if self.crop_duration >= self.segment_duration / 2:
            raise ValueError("crop_duration must be under half of segment_duration")
        if self.crop_samples < 3:
            raise ValueError("crop_duration spans fewer than three samples")
        return self

    @property
    def segment_samples(self) -> int:
        """Number of samples in the periodic segment."""
        return round(self.sampling_frequency * self.segment_duration)

    @property
    def crop_samples(self) -> int:
        """Odd number of samples in the cropped SNR series."""
        return 2 * int(self.crop_duration * self.sampling_frequency / 2) + 1

    def run(self, *, overwrite: bool = False) -> Path:
        """Write complete injection products, reusing a matching finished run."""
        return _run(self, overwrite=overwrite)


def load_inject_config(path: Path, **overrides: Any) -> InjectConfig:
    """Load injection TOML, resolving paths relative to the file."""
    return load_config(path, InjectConfig, **overrides)


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

    config: InjectConfig
    registry: DetectorRegistry
    population: Population


def load_inputs(config: InjectConfig) -> Inputs:
    """Read the detector registry and population a config points to.

    Parameters
    ----------
    config
        Run configuration.
    """
    from injstar import products
    from injstar.detector.network import load_detector_config
    from injstar.population import read_population

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


def _simulation_batches(inputs: Inputs) -> Iterator[InjectionResult]:
    """Configure JAX and yield results from one prepared simulation kernel."""
    from injstar.population import simulation_events
    from injstar.runtime import configure_runtime

    configure_runtime()
    from injstar.inject import _inject_batches, greenwich_sidereal_time
    from injstar.waveform import waveform

    config = inputs.config
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
        minimum_frequency=config.minimum_frequency,
        maximum_frequency=config.maximum_frequency,
        sampling_frequency=config.sampling_frequency,
        segment_duration=config.segment_duration,
        crop_duration=config.crop_duration,
        noise=config.noise,
        seed=config.seed,
        batch_size=config.batch_size,
        detector_snr=config.detection.detector_snr,
        network_snr=config.detection.network_snr,
        min_detectors=config.detection.min_detectors,
    )


def _run(config: InjectConfig, *, overwrite: bool = False) -> Path:
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
    import numpy as np

    from injstar import io

    inputs = load_inputs(config)
    target = config.output_dir / io.INJECTIONS_FILE
    reused = prepare_directory(config, overwrite=overwrite)
    results_path = config.output_dir / io.INJECTION_RESULTS_FILE
    if reused and target.is_file() and results_path.is_file():
        logger.info("reusing %s", target)
        return target

    write_config(config)
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
