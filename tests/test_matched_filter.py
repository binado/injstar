import jax.numpy as jnp
import numpy as np
import pytest

from injstar.matched_filter import (
    first_frequency_cumulant,
    matched_filter_snr_series,
    noise_weighted_inner_product,
    optimal_snr,
    second_frequency_cumulant,
)

DELTA_F = 0.5
N_SAMPLES = 1024


@pytest.fixture
def bins() -> np.ndarray:
    return np.arange(20, 200)


@pytest.fixture
def frequencies(bins: np.ndarray) -> np.ndarray:
    return bins * DELTA_F


@pytest.fixture
def amplitude(frequencies: np.ndarray) -> jnp.ndarray:
    return jnp.asarray(frequencies ** (-7 / 6))


@pytest.fixture
def inverse_psd(bins: np.ndarray) -> jnp.ndarray:
    return jnp.ones((2, len(bins)))


def test_inner_product_of_unit_template_is_four_df_per_unit_weight() -> None:
    h = jnp.ones(10)
    inverse_psd = jnp.full(10, 0.25)

    value = noise_weighted_inner_product(h, h, inverse_psd, delta_f=2.0)

    assert value == pytest.approx(4 * 2.0 * 10 * 0.25)


def test_inner_product_is_real_and_symmetric() -> None:
    a = jnp.array([1 + 2j, 3 - 1j, 0.5j])
    b = jnp.array([2 - 1j, 1j, 4 + 0j])
    inverse_psd = jnp.array([1.0, 2.0, 0.5])

    forward = noise_weighted_inner_product(a, b, inverse_psd, 1.0)
    backward = noise_weighted_inner_product(b, a, inverse_psd, 1.0)

    assert jnp.isrealobj(forward)
    assert forward == pytest.approx(backward)


def test_inner_product_ignores_bins_with_zero_inverse_psd() -> None:
    h = jnp.array([1.0, 5.0, 1.0])

    masked = noise_weighted_inner_product(h, h, jnp.array([1.0, 0.0, 1.0]), 1.0)

    assert masked == pytest.approx(4 * 2.0)


def test_optimal_snr_scales_linearly_with_the_template(
    amplitude: jnp.ndarray, inverse_psd: jnp.ndarray
) -> None:
    base = optimal_snr(amplitude, inverse_psd, DELTA_F)
    scaled = optimal_snr(3.0 * amplitude, inverse_psd, DELTA_F)

    np.testing.assert_allclose(scaled, 3.0 * base)


def test_optimal_snr_is_zero_without_weight(amplitude: jnp.ndarray) -> None:
    zero = jnp.zeros((1, len(amplitude)))

    assert optimal_snr(amplitude, zero, DELTA_F) == 0.0


def test_cumulants_of_a_single_line_are_its_frequency_and_zero_variance() -> None:
    frequencies = jnp.array([10.0, 20.0, 30.0])
    h = jnp.array([0.0, 1.0, 0.0])
    inverse_psd = jnp.ones(3)

    assert first_frequency_cumulant(frequencies, h, inverse_psd) == pytest.approx(
        2 * np.pi * 20.0
    )
    assert second_frequency_cumulant(frequencies, h, inverse_psd) == pytest.approx(0.0)


def test_cumulants_of_two_equal_lines_are_their_midpoint_and_half_spacing() -> None:
    frequencies = jnp.array([10.0, 30.0])
    h = jnp.ones(2)
    inverse_psd = jnp.ones(2)

    assert first_frequency_cumulant(frequencies, h, inverse_psd) == pytest.approx(
        2 * np.pi * 20.0
    )
    assert second_frequency_cumulant(frequencies, h, inverse_psd) == pytest.approx(
        (2 * np.pi * 10.0) ** 2
    )


def test_cumulants_are_zero_without_weight() -> None:
    frequencies = jnp.array([10.0, 20.0])
    h = jnp.ones(2)
    zero = jnp.zeros(2)

    assert first_frequency_cumulant(frequencies, h, zero) == 0.0
    assert second_frequency_cumulant(frequencies, h, zero) == 0.0


def test_series_peaks_at_the_peak_time_with_the_expected_snr(
    amplitude: jnp.ndarray, inverse_psd: jnp.ndarray, bins: np.ndarray
) -> None:
    rate = N_SAMPLES * DELTA_F
    peak_sample = np.array([100, 300])
    rho = jnp.array([8.0 * np.exp(0.7j), 5.0 * np.exp(-1.2j)])

    series = matched_filter_snr_series(
        amplitude, inverse_psd, rho, peak_sample / rate, bins, N_SAMPLES, DELTA_F
    )

    assert series.shape == (2, N_SAMPLES)
    np.testing.assert_array_equal(np.argmax(np.abs(series), axis=-1), peak_sample)
    np.testing.assert_allclose(series[[0, 1], peak_sample], rho, rtol=1e-10)


def test_series_without_noise_never_exceeds_the_expected_snr(
    amplitude: jnp.ndarray, inverse_psd: jnp.ndarray, bins: np.ndarray
) -> None:
    rho = jnp.array([8.0, 5.0], dtype=complex)

    series = matched_filter_snr_series(
        amplitude, inverse_psd, rho, jnp.array([0.1, 0.2]), bins, N_SAMPLES, DELTA_F
    )

    assert np.all(np.abs(series) <= np.abs(rho)[:, None] * (1 + 1e-10))


def test_series_noise_has_unit_variance_per_component(
    amplitude: jnp.ndarray, bins: np.ndarray
) -> None:
    # One row per independent noise draw; the signal is switched off by rho = 0.
    draws = 2000
    inverse_psd = jnp.ones((draws, len(bins)))
    rng = np.random.default_rng(0)
    white = rng.standard_normal((draws, len(bins))) + 1j * rng.standard_normal(
        (draws, len(bins))
    )

    series = matched_filter_snr_series(
        amplitude,
        inverse_psd,
        jnp.zeros(draws, dtype=complex),
        jnp.zeros(draws),
        bins,
        N_SAMPLES,
        DELTA_F,
        white,
    )

    np.testing.assert_allclose(np.std(series.real), 1.0, rtol=0.1)
    np.testing.assert_allclose(np.std(series.imag), 1.0, rtol=0.1)
