"""Run configuration.

A run is described by one TOML file, validated here. Paths in the file are
resolved against the file's directory. Nothing here imports JAX.

Example::

    network = ["H1", "L1", "V1"]
    population = "population.csv"
    output_dir = "out"
    minimum_frequency = 20.0
    maximum_frequency = 1024.0

    [prior]
    max_distance = 2000.0
"""

import tomllib
from pathlib import Path
from typing import Annotated, Any, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

_STRICT = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

Positive = Annotated[float, Field(gt=0)]


class DetectionConfig(BaseModel):
    """Detection rule applied to the observed SNRs.

    An event is detected when at least ``min_detectors`` detectors reach
    ``detector_snr`` and the network SNR reaches ``network_snr``.
    """

    model_config = _STRICT

    detector_snr: Annotated[float, Field(ge=0)] = 4.0
    network_snr: Annotated[float, Field(ge=0)] = 8.0
    min_detectors: Annotated[int, Field(ge=1)] = 2


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


class RunConfig(BaseModel):
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
    population_cosmology
        Name of an astropy cosmology, used only when the population gives
        source-frame masses and redshift.
    """

    model_config = _STRICT

    network: str | tuple[str, ...]
    detector_config: tuple[Path, ...] = ()
    population: Path
    output_dir: Path

    waveform_approximant: str = "IMRPhenomD"
    reference_frequency: Positive = 20.0
    minimum_frequency: Positive = 20.0
    maximum_frequency: Positive = 1024.0
    sampling_frequency: Positive = 4096.0
    segment_duration: Positive = 8.0
    crop_duration: Positive = 0.1
    noise: bool = True

    detection: DetectionConfig = DetectionConfig()
    prior: PriorConfig
    rescale_loglikelihood: Positive = 1.0
    population_cosmology: str = "Planck18"

    seed: Annotated[int, Field(ge=0)] = 0
    batch_size: Annotated[int, Field(ge=1)] = 32
    workers: Annotated[int, Field(ge=1)] = 1
    threads_per_worker: Annotated[int, Field(ge=1)] = 1

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


def load_run_config(path: Path, **overrides: Any) -> RunConfig:
    """Read and validate a run config.

    Relative paths in the file are resolved against the file's directory.

    Parameters
    ----------
    path
        TOML file.
    **overrides
        Top-level fields replacing those in the file; ``None`` values are
        ignored. Paths given here are resolved against the working directory.

    Raises
    ------
    pydantic.ValidationError
        If the file is not a valid run config.
    """
    path = Path(path)
    with path.open("rb") as handle:
        data = tomllib.load(handle)
    base = path.resolve().parent
    for key in ("population", "output_dir"):
        if key in data:
            data[key] = str(base / data[key])
    if "detector_config" in data:
        data["detector_config"] = [str(base / item) for item in data["detector_config"]]
    for key, value in overrides.items():
        if value is not None:
            data[key] = str(Path(value).resolve()) if key == "output_dir" else value
    return RunConfig.model_validate(data)
