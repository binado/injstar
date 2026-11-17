from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from injstar.config import InjectConfig
from injstar.dataset import InjectionResult
from injstar.detector.config import DetectorRegistry
from injstar.detector.network import DetectorGeometry
from injstar.detector.projection import Detector
from injstar.inject import greenwich_sidereal_time, inject_events
from injstar.localize import Event, Network
from injstar.population import read_population, simulation_events
from injstar.runtime import configure_runtime
from injstar.waveform import waveform
from injstar.waveform.pycbc import PyCBCWaveformGenerator

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
    return DetectorRegistry.load()


@pytest.fixture
def population_csv(tmp_path: Path) -> Path:
    path = tmp_path / "population.csv"
    path.write_text("\n".join((POPULATION_HEADER, *POPULATION_ROWS)) + "\n")
    return path


@pytest.fixture
def config_file(tmp_path: Path, population_csv: Path) -> Path:
    path = tmp_path / "inject.toml"
    path.write_text(
        'network = ["H1", "L1", "V1"]\n'
        f'population = "{population_csv.name}"\n'
        'output_dir = "out"\n'
        "segment_duration = 4.0\n"
        "batch_size = 2\n"
    )
    return path


def make_config(tmp_path: Path, **fields: object) -> InjectConfig:
    defaults: dict[str, object] = {
        "network": ("H1", "L1", "V1"),
        "population": tmp_path / "population.csv",
        "output_dir": tmp_path / "out",
        "segment_duration": 4.0,
    }
    return InjectConfig.model_validate(defaults | fields)


def inject(
    events: dict[str, np.ndarray],
    *,
    config: InjectConfig,
    network: DetectorRegistry,
    rows: np.ndarray | None = None,
) -> InjectionResult:
    """Inject using the explicit generators specified by the test configuration."""
    return inject_events(
        events,
        waveform=waveform(
            config.waveform_approximant, reference_frequency=config.reference_frequency
        ),
        template=waveform(
            config.template_approximant or config.waveform_approximant,
            reference_frequency=config.reference_frequency,
        ),
        network=network.select(config.network),
        minimum_frequency=config.minimum_frequency,
        maximum_frequency=config.maximum_frequency,
        sampling_frequency=config.sampling_frequency,
        segment_duration=config.segment_duration,
        crop_duration=config.crop_duration,
        noise=config.noise,
        seed=config.seed,
        batch_size=config.batch_size,
        **config.detection.model_dump(),
        rows=rows,
    )


@pytest.fixture
def event_and_network(
    tmp_path: Path, registry: DetectorRegistry, population_csv: Path
) -> tuple[Event, Network]:
    config = make_config(tmp_path, noise=False)
    population = read_population(population_csv)
    events = simulation_events(
        population, greenwich_sidereal_time(population["coa_time"])
    )
    drawn = inject(
        {name: values[:1] for name, values in events.items()},
        config=config,
        network=registry,
    )
    names, responses, locations = registry.arrays(config.network)
    event = Event(
        row_index=0,
        right_ascension=float(population["right_ascension"][0]),
        declination=float(population["declination"][0]),
        distance=float(population["distance"][0]),
        coa_time=float(population["coa_time"][0]),
        series=np.asarray(drawn.series[0]),
        epochs=np.asarray(drawn.epochs[0]),
        snr=np.asarray(drawn.observed_snr[0]),
        bandwidth=np.asarray(drawn.bandwidth[0]),
        horizon=np.asarray(drawn.horizon[0]),
    )
    network = Network(names, responses, locations, config.sampling_frequency)
    return event, network
