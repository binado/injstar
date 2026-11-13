from pathlib import Path

import lal
import lalsimulation
import numpy as np
import pytest

from injstar.detector.network import DetectorRegistry
from injstar.population import read_population, simulation_events
from injstar.simulate import Simulator, greenwich_sidereal_time

from .conftest import make_config


@pytest.fixture
def events(population_csv: Path) -> dict[str, np.ndarray]:
    population = read_population(population_csv)
    return simulation_events(
        population, greenwich_sidereal_time(population["coa_time"])
    )


def repeat(
    events: dict[str, np.ndarray], row: int, count: int
) -> dict[str, np.ndarray]:
    out = {
        name: np.repeat(values[row : row + 1], count) for name, values in events.items()
    }
    out["event_id"] = np.arange(count, dtype=np.uint64)
    return out


def test_zero_noise_peak_is_the_expected_snr(
    tmp_path: Path, registry: DetectorRegistry, events: dict[str, np.ndarray]
) -> None:
    simulator = Simulator.from_config(make_config(tmp_path, noise=False), registry)

    result = simulator.series(events)

    scalars = result.scalars
    # The peak falls between samples, so the sampled maximum is just below it.
    assert np.all(scalars.observed_snr <= scalars.expected_snr * (1 + 1e-9))
    np.testing.assert_allclose(scalars.observed_snr, scalars.expected_snr, rtol=0.02)
    centre = result.series.shape[-1] // 2
    np.testing.assert_allclose(
        np.abs(result.series[..., centre]), scalars.observed_snr, rtol=1e-12
    )
    assert result.series.shape == (3, 3, simulator.crop_samples)


def test_centre_sample_tracks_the_arrival_time(
    tmp_path: Path, registry: DetectorRegistry, events: dict[str, np.ndarray]
) -> None:
    config = make_config(tmp_path, noise=False)
    result = Simulator.from_config(config, registry).series(events)

    centre_time = result.epochs + (config.crop_samples // 2) / config.sample_rate
    for index, name in enumerate(("H1", "L1", "V1")):
        location = registry.detectors[name].geometry.location
        delay = lal.TimeDelayFromEarthCenter(
            location, 1.2, -0.4, lal.LIGOTimeGPS(1400000000)
        )
        assert abs(centre_time[0, index] - delay) <= 0.5 / config.sample_rate + 1e-9


def test_horizon_matches_lal_amplitude(
    tmp_path: Path, registry: DetectorRegistry, events: dict[str, np.ndarray]
) -> None:
    simulator = Simulator.from_config(make_config(tmp_path, noise=False), registry)
    horizon = simulator.snr(events).horizon[0, 0]

    plus, _ = lalsimulation.SimInspiralChooseFDWaveform(
        36.0 * lal.MSUN_SI, 29.0 * lal.MSUN_SI, 0, 0, 0, 0, 0, 0,
        1e6 * lal.PC_SI, 0, 0, 0, 0, 0,
        simulator.delta_f, 20.0, 1024.0, 20.0, None, lalsimulation.IMRPhenomD,
    )  # fmt: skip
    bins = np.round(simulator.frequencies / simulator.delta_f).astype(int)
    psd = registry.detectors["H1"].sensitivity(simulator.frequencies)
    expected = np.sqrt(
        4 * simulator.delta_f * np.sum(np.abs(plus.data.data[bins]) ** 2 / psd)
    )

    assert horizon == pytest.approx(expected, rel=1e-6)


def test_noise_has_unit_variance_per_quadrature(
    tmp_path: Path, registry: DetectorRegistry, events: dict[str, np.ndarray]
) -> None:
    many = repeat(events, 0, 400)
    clean = Simulator.from_config(make_config(tmp_path, noise=False), registry).series(
        repeat(events, 0, 1)
    )
    noisy = Simulator.from_config(
        make_config(tmp_path, batch_size=100), registry
    ).series(many)

    residual = noisy.series - clean.series[0]
    centre = residual.shape[-1] // 2

    # 400 draws: the sample variance scatters by about 7%.
    np.testing.assert_allclose(residual[..., centre].real.var(axis=0), 1.0, rtol=0.3)
    np.testing.assert_allclose(residual[..., centre].imag.var(axis=0), 1.0, rtol=0.3)
    assert abs(residual[..., centre].mean()) < 0.2


def test_results_do_not_depend_on_batching_or_order(
    tmp_path: Path, registry: DetectorRegistry, events: dict[str, np.ndarray]
) -> None:
    many = repeat(events, 0, 7)
    order = np.array([3, 0, 6, 1, 5, 2, 4])
    reference = Simulator.from_config(
        make_config(tmp_path, batch_size=7), registry
    ).series(many)
    shuffled = Simulator.from_config(
        make_config(tmp_path, batch_size=3), registry
    ).series({name: values[order] for name, values in many.items()})

    np.testing.assert_allclose(
        shuffled.series, reference.series[order], rtol=0, atol=1e-10
    )
    np.testing.assert_array_equal(
        shuffled.scalars.detected, reference.scalars.detected[order]
    )


def test_seed_changes_the_noise(
    tmp_path: Path, registry: DetectorRegistry, events: dict[str, np.ndarray]
) -> None:
    first = Simulator.from_config(make_config(tmp_path, seed=1), registry).snr(events)
    second = Simulator.from_config(make_config(tmp_path, seed=2), registry).snr(events)

    assert not np.allclose(first.observed_snr, second.observed_snr)
    np.testing.assert_array_equal(first.expected_snr, second.expected_snr)


def test_detection_rule(
    tmp_path: Path, registry: DetectorRegistry, events: dict[str, np.ndarray]
) -> None:
    result = Simulator.from_config(make_config(tmp_path), registry).snr(events)

    # Loud, far too distant, and a nearby binary neutron star.
    np.testing.assert_array_equal(result.detected, [True, False, True])
    np.testing.assert_allclose(
        result.network_snr, np.sqrt(np.sum(result.observed_snr**2, axis=1))
    )
