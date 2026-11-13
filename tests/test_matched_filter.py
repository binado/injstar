import jax.numpy as jnp
import numpy as np
import pytest

from injstar.matched_filter import (
    band_bins,
    first_frequency_cumulant,
    matched_filter_snr_series,
    noise_weighted_inner_product,
    optimal_snr,
    second_frequency_cumulant,
)

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


@pytest.fixture
def bins() -> np.ndarray:
    return band_bins(**BAND)


@pytest.fixture
def frequencies(bins: np.ndarray) -> np.ndarray:
    return bins * DELTA_F


@pytest.fixture
def template(frequencies: np.ndarray) -> jnp.ndarray:
    return jnp.asarray(frequencies ** (-7 / 6))


@pytest.fixture
def inverse_psd(bins: np.ndarray) -> jnp.ndarray:
    return jnp.ones((2, len(bins)))


def injected(
    template: jnp.ndarray,
    frequencies: np.ndarray,
    rho: jnp.ndarray,
    peak_time: jnp.ndarray,
    inverse_psd: jnp.ndarray,
) -> jnp.ndarray:
    """Data holding the template with matched-filter SNR ``rho`` at ``peak_time``."""
    sigma = optimal_snr(template, inverse_psd=inverse_psd, delta_f=DELTA_F)
    phase = jnp.exp(-2j * jnp.pi * frequencies * peak_time[:, None])
    return (rho / sigma)[:, None] * template * phase


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


def test_inner_product_of_unit_template_is_four_df_per_unit_weight() -> None:
    template = jnp.ones(10)

    value = noise_weighted_inner_product(
        template, template, inverse_psd=jnp.full(10, 0.25), delta_f=2.0
    )

    assert value == pytest.approx(4 * 2.0 * 10 * 0.25)


def test_inner_product_is_real_and_symmetric() -> None:
    a = jnp.array([1 + 2j, 3 - 1j, 0.5j])
    b = jnp.array([2 - 1j, 1j, 4 + 0j])
    inverse_psd = jnp.array([1.0, 2.0, 0.5])

    forward = noise_weighted_inner_product(a, b, inverse_psd=inverse_psd, delta_f=1.0)
    backward = noise_weighted_inner_product(b, a, inverse_psd=inverse_psd, delta_f=1.0)

    assert jnp.isrealobj(forward)
    assert forward == pytest.approx(backward)


def test_inner_product_ignores_bins_with_zero_inverse_psd() -> None:
    template = jnp.array([1.0, 5.0, 1.0])

    masked = noise_weighted_inner_product(
        template, template, inverse_psd=jnp.array([1.0, 0.0, 1.0]), delta_f=1.0
    )

    assert masked == pytest.approx(4 * 2.0)


def test_optimal_snr_scales_linearly_with_the_template(
    template: jnp.ndarray, inverse_psd: jnp.ndarray
) -> None:
    base = optimal_snr(template, inverse_psd=inverse_psd, delta_f=DELTA_F)
    scaled = optimal_snr(3.0 * template, inverse_psd=inverse_psd, delta_f=DELTA_F)

    np.testing.assert_allclose(scaled, 3.0 * base)


def test_optimal_snr_is_zero_without_weight(template: jnp.ndarray) -> None:
    zero = jnp.zeros((1, len(template)))

    assert optimal_snr(template, inverse_psd=zero, delta_f=DELTA_F) == 0.0


def test_cumulants_of_a_single_line_are_its_frequency_and_zero_variance() -> None:
    kwargs = {"frequencies": jnp.array([10.0, 20.0, 30.0]), "inverse_psd": jnp.ones(3)}
    template = jnp.array([0.0, 1.0, 0.0])

    assert first_frequency_cumulant(template, **kwargs) == pytest.approx(
        2 * np.pi * 20.0
    )
    assert second_frequency_cumulant(template, **kwargs) == pytest.approx(0.0)


def test_cumulants_of_two_equal_lines_are_their_midpoint_and_half_spacing() -> None:
    kwargs = {"frequencies": jnp.array([10.0, 30.0]), "inverse_psd": jnp.ones(2)}
    template = jnp.ones(2)

    assert first_frequency_cumulant(template, **kwargs) == pytest.approx(
        2 * np.pi * 20.0
    )
    assert second_frequency_cumulant(template, **kwargs) == pytest.approx(
        (2 * np.pi * 10.0) ** 2
    )


def test_cumulants_are_zero_without_weight() -> None:
    kwargs = {"frequencies": jnp.array([10.0, 20.0]), "inverse_psd": jnp.zeros(2)}
    template = jnp.ones(2)

    assert first_frequency_cumulant(template, **kwargs) == 0.0
    assert second_frequency_cumulant(template, **kwargs) == 0.0


def test_series_of_injected_template_peaks_at_its_time_with_its_snr(
    template: jnp.ndarray,
    inverse_psd: jnp.ndarray,
    frequencies: np.ndarray,
    bins: np.ndarray,
) -> None:
    peak_sample = np.array([100, 300])
    rho = jnp.array([8.0 * np.exp(0.7j), 5.0 * np.exp(-1.2j)])
    data = injected(
        template, frequencies, rho, peak_sample / SAMPLING_FREQUENCY, inverse_psd
    )

    series = matched_filter_snr_series(
        data,
        template,
        inverse_psd=inverse_psd,
        **BAND,
    )

    assert series.shape == (2, N_SAMPLES)
    np.testing.assert_array_equal(np.argmax(np.abs(series), axis=-1), peak_sample)
    np.testing.assert_allclose(series[[0, 1], peak_sample], rho, rtol=1e-10)


def test_series_is_linear_in_the_data(
    template: jnp.ndarray,
    inverse_psd: jnp.ndarray,
    frequencies: np.ndarray,
    bins: np.ndarray,
) -> None:
    rho = jnp.array([8.0, 5.0], dtype=complex)
    data = injected(template, frequencies, rho, jnp.array([0.1, 0.2]), inverse_psd)

    single = matched_filter_snr_series(data, template, inverse_psd=inverse_psd, **BAND)
    double = matched_filter_snr_series(
        2 * data, template, inverse_psd=inverse_psd, **BAND
    )

    np.testing.assert_allclose(double, 2 * single)


def test_series_ignores_data_in_bins_without_a_psd(
    template: jnp.ndarray, bins: np.ndarray
) -> None:
    inverse_psd = jnp.ones((1, len(bins))).at[:, :10].set(0.0)
    clean = jnp.ones((1, len(bins)), dtype=complex)
    dirty = clean.at[:, :10].set(1e6)

    expected = matched_filter_snr_series(
        clean, template, inverse_psd=inverse_psd, **BAND
    )
    result = matched_filter_snr_series(dirty, template, inverse_psd=inverse_psd, **BAND)

    np.testing.assert_allclose(result, expected)


def test_series_of_noise_has_unit_variance_per_component(
    template: jnp.ndarray, bins: np.ndarray
) -> None:
    # Gaussian noise of one-sided PSD S has per-bin variance S / (4 df), so it
    # matched-filters to unit variance per real and imaginary component.
    draws = 2000
    psd = 2.0 + np.linspace(0.0, 1.0, len(bins))
    rng = np.random.default_rng(0)
    white = rng.standard_normal((draws, len(bins))) + 1j * rng.standard_normal(
        (draws, len(bins))
    )
    noise = np.sqrt(psd / (4 * DELTA_F)) * white

    series = matched_filter_snr_series(
        noise,
        template,
        inverse_psd=jnp.broadcast_to(1 / psd, (draws, len(bins))),
        **BAND,
    )

    np.testing.assert_allclose(np.std(series.real), 1.0, rtol=0.1)
    np.testing.assert_allclose(np.std(series.imag), 1.0, rtol=0.1)


def test_series_rejects_arrays_that_do_not_match_the_band(
    template: jnp.ndarray, inverse_psd: jnp.ndarray
) -> None:
    short = jnp.ones((2, len(template) - 1), dtype=complex)

    with pytest.raises(ValueError, match="bins"):
        matched_filter_snr_series(
            short, template[:-1], inverse_psd=inverse_psd[:, :-1], **BAND
        )
