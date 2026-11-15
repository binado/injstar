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
from injstar.waveform.pycbc import PyCBCWaveformGenerator

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


class PycbcDetector(Detector[NDArray[np.float64]]):
    """PyCBC projection with an explicit sidereal angle."""

    def __init__(self, name: str, *, geometry: DetectorGeometry | None = None) -> None:
        from pycbc.detector import Detector as PycbcGroundDetector

        class SiderealDetector(PycbcGroundDetector):
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
def pycbc_waveform() -> type[PyCBCWaveformGenerator]:
    """Factory for a PyCBC waveform with the requested model and reference frequency."""
    return PyCBCWaveformGenerator


@pytest.fixture(scope="session")
def pycbc_detector() -> type[PycbcDetector]:
    """Factory for a named PyCBC detector, optionally with injstar geometry."""
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
        row_index=0,
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
