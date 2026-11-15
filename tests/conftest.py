from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from injstar.config import RunConfig
from injstar.detector.network import (
    DetectorGeometry,
    DetectorRegistry,
    load_detector_config,
)
from injstar.detector.projection import Detector
from injstar.localize import Event, Network
from injstar.population import read_population, simulation_events
from injstar.runtime import configure_runtime
from injstar.simulate import Simulator, greenwich_sidereal_time
from injstar.waveform import WaveformGenerator

# 64-bit JAX has to be chosen before any test touches the backend.
configure_runtime()

POPULATION_HEADER = (
    "detector_frame_mass_1,detector_frame_mass_2,spin_1z,spin_2z,distance,"
    "right_ascension,declination,inclination,polarization_angle,coa_time"
)
POPULATION_ROWS = (
    "36.0,29.0,0.0,0.0,1500.0,1.2,-0.4,0.5,0.3,1400000000.0",
    "30.0,25.0,0.1,-0.1,40000.0,2.0,0.6,1.2,1.0,1400000100.5",
    "1.4,1.3,0.0,0.0,200.0,4.0,-1.0,2.6,0.1,1400000200.25",
)


@dataclass(frozen=True)
class PycbcWaveform(WaveformGenerator[np.ndarray]):
    """PyCBC's aligned-spin IMRPhenomD reference model."""

    reference_frequency: float

    @property
    def parameter_names(self) -> tuple[str, ...]:
        return (
            "detector_frame_mass_1",
            "detector_frame_mass_2",
            "spin_1z",
            "spin_2z",
            "distance",
            "coa_phase",
            "inclination",
        )

    def __call__(
        self, frequencies: np.ndarray, source: Mapping[str, np.ndarray | float]
    ) -> tuple[NDArray[np.complex128], NDArray[np.complex128]]:
        from pycbc.waveform import get_fd_waveform

        if frequencies.ndim != 1 or len(frequencies) < 2:
            raise ValueError("expected a one-dimensional grid of at least two bins")
        delta_f = float(frequencies[-1] - frequencies[0]) / (len(frequencies) - 1)
        if delta_f <= 0 or frequencies[0] <= 0 or not np.isfinite(frequencies).all():
            raise ValueError("expected finite, positive, increasing frequencies")
        bin_numbers = frequencies / delta_f
        bins = np.rint(bin_numbers).astype(np.intp)
        if not np.allclose(bin_numbers, bins, rtol=0, atol=1e-7) or not np.all(
            np.diff(bins) == 1
        ):
            raise ValueError("expected contiguous bins on a zero-origin FFT grid")

        plus, cross = get_fd_waveform(
            approximant="IMRPhenomD",
            mass1=float(source["detector_frame_mass_1"]),
            mass2=float(source["detector_frame_mass_2"]),
            spin1z=float(source["spin_1z"]),
            spin2z=float(source["spin_2z"]),
            distance=float(source["distance"]),
            coa_phase=float(source["coa_phase"]),
            inclination=float(source["inclination"]),
            delta_f=delta_f,
            f_lower=float(frequencies[0]),
            f_final=float(frequencies[-1] + delta_f),
            f_ref=self.reference_frequency,
        )
        # Use the FD samples directly. The series epoch describes the FFT
        # segment and is not an extra coalescence-time shift.
        return (
            np.asarray(plus.numpy(), dtype=np.complex128)[bins],
            np.asarray(cross.numpy(), dtype=np.complex128)[bins],
        )


class PycbcDetector(Detector[NDArray[np.float64]]):
    """PyCBC projection with an explicit sidereal angle."""

    def __init__(self, name: str, *, geometry: DetectorGeometry | None = None) -> None:
        from pycbc.detector import Detector as GroundDetector

        class SiderealDetector(GroundDetector):
            def gmst_estimate(
                self, gps_time: NDArray[np.float64]
            ) -> NDArray[np.float64]:
                # The projection entry points pass their time argument through
                # here. Supply GMST directly to use the same angle as injstar.
                return gps_time

        self._detector = SiderealDetector(name)
        if geometry is not None:
            self._detector.response = geometry.response_tensor
            self._detector.location = geometry.location

    def response_tensor(self) -> NDArray[np.float64]:
        return np.asarray(self._detector.response, dtype=np.float64)

    def location(self) -> NDArray[np.float64]:
        return np.asarray(self._detector.location, dtype=np.float64)

    def antenna_pattern(
        self,
        gmst: NDArray[np.float64] | float,
        *,
        right_ascension: NDArray[np.float64] | float,
        declination: NDArray[np.float64] | float,
        polarization_angle: NDArray[np.float64] | float,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        ra, dec, psi, sidereal = np.broadcast_arrays(
            right_ascension, declination, polarization_angle, gmst
        )
        plus, cross = self._detector.antenna_pattern(
            ra.ravel(), dec.ravel(), psi.ravel(), sidereal.ravel()
        )
        return (
            np.asarray(plus, dtype=np.float64).reshape(ra.shape),
            np.asarray(cross, dtype=np.float64).reshape(ra.shape),
        )

    def time_delay_from_geocenter(
        self,
        gmst: NDArray[np.float64] | float,
        *,
        right_ascension: NDArray[np.float64] | float,
        declination: NDArray[np.float64] | float,
    ) -> NDArray[np.float64]:
        ra, dec, sidereal = np.broadcast_arrays(right_ascension, declination, gmst)
        delay = self._detector.time_delay_from_earth_center(
            ra.ravel(), dec.ravel(), sidereal.ravel()
        )
        return np.asarray(delay, dtype=np.float64).reshape(ra.shape)


@pytest.fixture(scope="session")
def pycbc_waveform() -> type[PycbcWaveform]:
    """Factory for a PyCBC waveform with the requested reference frequency."""
    pytest.importorskip("pycbc.waveform")
    return PycbcWaveform


@pytest.fixture(scope="session")
def pycbc_detector() -> type[PycbcDetector]:
    """Factory for a named PyCBC detector, optionally with injstar geometry."""
    pytest.importorskip("pycbc.detector")
    return PycbcDetector


@pytest.fixture(scope="session")
def registry() -> DetectorRegistry:
    return load_detector_config()


@pytest.fixture
def population_csv(tmp_path: Path) -> Path:
    path = tmp_path / "population.csv"
    path.write_text("\n".join((POPULATION_HEADER, *POPULATION_ROWS)) + "\n")
    return path


@pytest.fixture
def config_file(tmp_path: Path, population_csv: Path) -> Path:
    path = tmp_path / "run.toml"
    path.write_text(
        'network = ["H1", "L1", "V1"]\n'
        f'population = "{population_csv.name}"\n'
        'output_dir = "out"\n'
        "segment_duration = 4.0\n"
        "batch_size = 2\n"
        "[prior]\n"
        "max_distance = 5000.0\n"
    )
    return path


def make_config(tmp_path: Path, **fields: object) -> RunConfig:
    defaults: dict[str, object] = {
        "network": ("H1", "L1", "V1"),
        "population": tmp_path / "population.csv",
        "output_dir": tmp_path / "out",
        "segment_duration": 4.0,
        "prior": {"max_distance": 5000.0},
    }
    return RunConfig.model_validate(defaults | fields)


@pytest.fixture
def event_and_network(
    tmp_path: Path, registry: DetectorRegistry, population_csv: Path
) -> tuple[Event, Network]:
    config = make_config(tmp_path, noise=False)
    population = read_population(population_csv)
    events = simulation_events(
        population, greenwich_sidereal_time(population["coa_time"])
    )
    drawn = Simulator.from_config(config, registry).series(
        {name: values[:1] for name, values in events.items()}
    )
    names, responses, locations = registry.arrays(config.network)
    event = Event(
        event_id=0,
        coa_time=float(population["coa_time"][0]),
        series=drawn.series[0],
        epochs=drawn.epochs[0],
        snr=drawn.scalars.observed_snr[0],
        bandwidth=drawn.scalars.bandwidth[0],
        horizon=drawn.scalars.horizon[0],
    )
    network = Network(
        names, responses, locations, config.sampling_frequency, config.prior, 1.0
    )
    return event, network
