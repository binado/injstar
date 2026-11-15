import jax.numpy as jnp
import numpy as np
import pytest
from pycbc import filter as pycbc_filter
from pycbc.noise import frequency_noise_from_psd
from pycbc.types import FrequencySeries

from injstar.matched_filter import (
    band_bins,
    first_frequency_cumulant,
    matched_filter_snr_series,
    noise_weighted_inner_product,
    optimal_snr,
    second_frequency_cumulant,
)

from .tolerances import assert_close

SEGMENT_DURATION = 2.0
SAMPLING_FREQUENCY = 512.0
DELTA_F = 1 / SEGMENT_DURATION
N_SAMPLES = 1024
BAND = {
    "minimum_frequency": 10.0,
    "maximum_frequency": 99.5,
    "sampling_frequency": SAMPLING_FREQUENCY,
    "segment_duration": SEGMENT_DURATION,
}

REFERENCE_BAND = BAND | {"minimum_frequency": 10.1, "maximum_frequency": 99.9}
REFERENCE_BINS = slice(21, 200)
REFERENCE_CUTOFFS = {"low_frequency_cutoff": 10.5, "high_frequency_cutoff": 100.0}


@pytest.fixture
def reference_spectra() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    frequencies = np.arange(N_SAMPLES // 2 + 1) * DELTA_F
    phase = 0.2 * np.sqrt(frequencies)
    template = np.exp(1j * phase)
    other = np.stack((-template + 0.3j, 0.7j * template + 0.1))
    psd = np.stack((1 + frequencies / 80, 3 + (frequencies / 60) ** 2))
    psd[:, 64] = np.inf
    return template, other, psd


def frequency_series(values: np.ndarray) -> FrequencySeries:
    return FrequencySeries(values, delta_f=DELTA_F)


def test_inner_product_and_norm_match_pycbc(
    reference_spectra: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    template, other, psd = reference_spectra
    h = frequency_series(template)
    overlaps = []
    norms = []
    for data, row in zip(other, psd, strict=True):
        weights = frequency_series(row)
        overlaps.append(
            pycbc_filter.overlap(
                h,
                frequency_series(data),
                psd=weights,
                normalized=False,
                **REFERENCE_CUTOFFS,
            )
        )
        norms.append(pycbc_filter.sigma(h, psd=weights, **REFERENCE_CUTOFFS))

    inverse_psd = 1 / psd[:, REFERENCE_BINS]
    band_template = template[REFERENCE_BINS]
    inner = noise_weighted_inner_product(
        band_template,
        other[:, REFERENCE_BINS],
        inverse_psd=inverse_psd,
        delta_f=DELTA_F,
    )
    assert jnp.isrealobj(inner)
    assert_close(inner, overlaps)
    assert_close(
        optimal_snr(band_template, inverse_psd=inverse_psd, delta_f=DELTA_F), norms
    )


def test_frequency_cumulants_match_pycbc(
    reference_spectra: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    template, _, psd = reference_spectra
    frequencies = np.arange(template.size) * DELTA_F
    h = frequency_series(template)
    omega_h = frequency_series(2 * np.pi * frequencies * template)
    centroids = []
    variances = []
    for row in psd:
        kwargs = {"psd": frequency_series(row), **REFERENCE_CUTOFFS}
        power = pycbc_filter.sigmasq(h, **kwargs)
        centroid = pycbc_filter.overlap(h, omega_h, normalized=False, **kwargs) / power
        centroids.append(centroid)
        variances.append(pycbc_filter.sigmasq(omega_h, **kwargs) / power - centroid**2)

    band_template = template[REFERENCE_BINS]
    band_frequencies = frequencies[REFERENCE_BINS]
    inverse_psd = 1 / psd[:, REFERENCE_BINS]
    for cumulant, expected in (
        (first_frequency_cumulant, centroids),
        (second_frequency_cumulant, variances),
    ):
        result = cumulant(
            band_template, frequencies=band_frequencies, inverse_psd=inverse_psd
        )
        assert_close(result, expected)


def test_complex_snr_series_matches_pycbc(
    reference_spectra: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    template, other, psd = reference_spectra
    frequencies = np.arange(template.size) * DELTA_F
    lag = -27.3 / SAMPLING_FREQUENCY
    shifted = template * np.exp(1j * (0.7 - 2 * np.pi * frequencies * lag))

    # PyCBC's noise generator uses NumPy's global RNG. Its zero PSD bins draw
    # zero noise; the filtering PSD uses infinity to give those bins zero weight.
    rng_state = np.random.get_state()
    try:
        noise = frequency_noise_from_psd(
            frequency_series(np.where(np.isfinite(psd[1]), psd[1], 0.0)), seed=24
        ).numpy()
    finally:
        np.random.set_state(rng_state)
    data = np.stack((shifted, other[1] + noise))
    h = frequency_series(template)
    expected = np.stack(
        [
            pycbc_filter.matched_filter(
                h, frequency_series(d), psd=frequency_series(row), **REFERENCE_CUTOFFS
            ).numpy()
            for d, row in zip(data, psd, strict=True)
        ]
    )
    inverse_psd = 1 / psd[:, REFERENCE_BINS]
    band_template = template[REFERENCE_BINS]
    sigma = optimal_snr(band_template, inverse_psd=inverse_psd, delta_f=DELTA_F)
    result = matched_filter_snr_series(
        data[:, REFERENCE_BINS],
        band_template / sigma[:, None],
        inverse_psd=inverse_psd,
        **REFERENCE_BAND,
    )

    assert result.shape == (2, N_SAMPLES)
    assert_close(result, expected)


@pytest.mark.parametrize(
    ("overrides", "first", "last"),
    [
        ({}, 20, 199),
        ({"minimum_frequency": 10.1}, 21, 199),
        ({"maximum_frequency": 99.9}, 20, 199),
        ({"maximum_frequency": 1000.0}, 20, N_SAMPLES // 2 - 1),
        ({"segment_duration": 4.0, "sampling_frequency": 256.0}, 40, 398),
    ],
)
def test_band_bins_span_the_band_on_the_segment_grid(
    overrides: dict[str, float], first: int, last: int
) -> None:
    bins = band_bins(**(BAND | overrides))

    np.testing.assert_array_equal(bins, np.arange(first, last + 1))


def test_band_bins_are_empty_for_a_band_between_bins() -> None:
    bins = band_bins(**(BAND | {"minimum_frequency": 10.1, "maximum_frequency": 10.4}))

    assert len(bins) == 0


def test_optimal_snr_is_zero_without_weight() -> None:
    assert optimal_snr(jnp.ones(3), inverse_psd=jnp.zeros(3), delta_f=DELTA_F) == 0.0


def test_cumulants_are_zero_without_weight() -> None:
    kwargs = {"frequencies": jnp.array([10.0, 20.0]), "inverse_psd": jnp.zeros(2)}
    template = jnp.ones(2)

    assert first_frequency_cumulant(template, **kwargs) == 0.0
    assert second_frequency_cumulant(template, **kwargs) == 0.0


def test_series_rejects_arrays_that_do_not_match_the_band() -> None:
    short = jnp.ones((2, 179), dtype=complex)
    template = jnp.ones(180)
    inverse_psd = jnp.ones((2, 180))

    with pytest.raises(ValueError, match="data has 179 bins but the band holds 180"):
        matched_filter_snr_series(short, template, inverse_psd=inverse_psd, **BAND)
