from pathlib import Path

import numpy as np
import pytest
from pycbc import filter as pycbc_filter
from pycbc.types import FrequencySeries

from injstar.detector.network import DetectorRegistry
from injstar.population import read_population, simulation_events
from injstar.simulate import Simulator, greenwich_sidereal_time
from injstar.waveform.pycbc import PyCBCWaveformGenerator

from .conftest import PycbcDetector, make_config


@pytest.fixture
def events(population_csv: Path) -> dict[str, np.ndarray]:
    population = read_population(population_csv)
    return simulation_events(
        population, greenwich_sidereal_time(population["coa_time"])
    )


@pytest.fixture
def tidal_events(population_csv: Path, tmp_path: Path) -> dict[str, np.ndarray]:
    header, *rows = population_csv.read_text().splitlines()
    path = tmp_path / "tidal.csv"
    path.write_text(
        "\n".join(
            [f"{header},lambda_1,lambda_2", *(f"{row},400.0,400.0" for row in rows)]
        )
        + "\n"
    )
    population = read_population(path)
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


@pytest.mark.parametrize("template_approximant", ["IMRPhenomD", "IMRPhenomXAS"])
def test_noise_free_series_matches_pycbc(
    tmp_path: Path,
    registry: DetectorRegistry,
    events: dict[str, np.ndarray],
    pycbc_waveform: type[PyCBCWaveformGenerator],
    pycbc_detector: type[PycbcDetector],
    template_approximant: str,
) -> None:
    config = make_config(
        tmp_path,
        noise=False,
        waveform_approximant="IMRPhenomD",
        template_approximant=template_approximant,
        segment_duration=2.0,
        sampling_frequency=1024.0,
        minimum_frequency=20.0,
        maximum_frequency=400.0,
        crop_duration=0.1,
    )
    event = {field: values[:1] for field, values in events.items()}
    event["coa_phase"] = np.array([0.6])
    result = Simulator.from_config(config, registry).series(event)

    source = {field: float(values[0]) for field, values in event.items()}
    bins = slice(40, 801)
    frequencies = np.arange(bins.start, bins.stop) * 0.5
    midpoint = config.segment_samples // 2
    half_crop = config.crop_samples // 2
    rate = config.sampling_frequency

    def frequency_series(values: np.ndarray, fill: float = 0.0) -> FrequencySeries:
        full = np.full(midpoint + 1, fill, dtype=values.dtype)
        full[bins] = values
        return FrequencySeries(full, delta_f=0.5)

    plus, cross = pycbc_waveform(config.waveform_approximant, 20.0)(frequencies, source)
    template, _ = pycbc_waveform(template_approximant, 20.0)(
        frequencies, source | {"distance": 1.0, "inclination": 0.0, "coa_phase": 0.0}
    )
    h = frequency_series(template)
    sky = {
        "right_ascension": source["right_ascension"],
        "declination": source["declination"],
    }
    # With PyCBC 2.11.0, the mixed-model crop differed by up to 2.6e-4
    # of peak SNR, mostly from phase differences in the compiled Ripple
    # injection. The 1e-3 tolerance covers that difference. With a matched
    # template, those phase differences mostly cancel and the crop agreed
    # within 4e-7 of peak SNR.
    series_tolerance = (
        1e-6 if config.waveform_approximant == template_approximant else 1e-3
    )
    assert result.series.shape == (1, len(config.network), config.crop_samples)
    for index, name in enumerate(config.network):
        registered = registry.detectors[name]
        detector = pycbc_detector(name, geometry=registered.geometry)
        fp, fc = detector.antenna_pattern(
            source["gmst"], polarization_angle=source["polarization_angle"], **sky
        )
        delay = float(detector.time_delay_from_geocenter(source["gmst"], **sky))
        strain = (fp * plus + fc * cross) * np.exp(
            -2j * np.pi * frequencies * (midpoint / rate + delay)
        )
        d = frequency_series(strain)
        kwargs = {
            "psd": frequency_series(registered.sensitivity(frequencies), fill=np.inf),
            "low_frequency_cutoff": 20.0,
            "high_frequency_cutoff": 400.5,
        }
        reference = pycbc_filter.matched_filter(h, d, **kwargs).numpy()
        start = midpoint + round(delay * rate) - half_crop
        crop = reference[start : start + config.crop_samples]
        peak = np.max(np.abs(crop))

        assert result.epochs[0, index] == (start - midpoint) / rate
        np.testing.assert_allclose(
            result.series[0, index] / peak,
            crop / peak,
            rtol=series_tolerance,
            atol=series_tolerance,
        )
        np.testing.assert_allclose(
            result.scalars.observed_snr[0, index], peak, rtol=series_tolerance
        )
        # The optimal SNR and horizon use waveform norms. Both agreed to
        # about 2e-7 relative across the two model pairs.
        np.testing.assert_allclose(
            [
                result.scalars.expected_snr[0, index],
                result.scalars.horizon[0, index],
            ],
            [pycbc_filter.sigma(d, **kwargs), pycbc_filter.sigma(h, **kwargs)],
            rtol=1e-6,
        )


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


@pytest.mark.parametrize(
    "approximant", ["IMRPhenomD_NRTidalv2", "IMRPhenomXAS_NRTidalv3", "TaylorF2"]
)
def test_tidal_model_gives_finite_positive_horizons(
    tmp_path: Path,
    registry: DetectorRegistry,
    tidal_events: dict[str, np.ndarray],
    approximant: str,
) -> None:
    config = make_config(tmp_path, noise=False, waveform_approximant=approximant)

    horizon = Simulator.from_config(config, registry).snr(tidal_events).horizon

    # The last row is the binary neutron star.
    assert np.all(np.isfinite(horizon[2]))
    assert np.all(horizon[2] > 0)


def test_tidal_deformability_reaches_a_tidal_model(
    tmp_path: Path,
    registry: DetectorRegistry,
    events: dict[str, np.ndarray],
    tidal_events: dict[str, np.ndarray],
) -> None:
    simulator = Simulator.from_config(
        make_config(tmp_path, noise=False, waveform_approximant="TaylorF2"), registry
    )

    undeformed = simulator.snr(events).horizon[2]
    deformed = simulator.snr(tidal_events).horizon[2]

    assert not np.array_equal(undeformed, deformed)


def test_event_fields_follow_the_waveform(
    tmp_path: Path, registry: DetectorRegistry
) -> None:
    aligned = Simulator.from_config(make_config(tmp_path), registry)
    tidal = Simulator.from_config(
        make_config(tmp_path, waveform_approximant="TaylorF2"), registry
    )

    assert "lambda_1" not in aligned.event_fields
    assert "lambda_1" in tidal.event_fields
    assert len(set(tidal.event_fields)) == len(tidal.event_fields)


@pytest.mark.parametrize("signal", ["IMRPhenomXHM", "IMRPhenomXPHM"])
def test_higher_mode_signal_never_exceeds_its_optimal_snr(
    tmp_path: Path,
    registry: DetectorRegistry,
    events: dict[str, np.ndarray],
    signal: str,
) -> None:
    config = make_config(
        tmp_path,
        noise=False,
        waveform_approximant=signal,
        template_approximant="IMRPhenomXAS",
    )

    result = Simulator.from_config(config, registry).snr(events)

    assert np.all(np.isfinite(result.expected_snr))
    assert np.all(np.isfinite(result.observed_snr))
    assert np.all(result.observed_snr <= result.expected_snr * (1 + 1e-9))
    # The dominant-mode template recovers most of a higher-mode signal.
    np.testing.assert_allclose(result.observed_snr, result.expected_snr, rtol=0.05)


def test_from_config_rejects_a_higher_mode_template(
    tmp_path: Path, registry: DetectorRegistry
) -> None:
    config = make_config(tmp_path, waveform_approximant="IMRPhenomXHM")

    with pytest.raises(ValueError, match="template_approximant"):
        Simulator.from_config(config, registry)
