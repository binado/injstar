r"""Matched-filter quantities of a frequency-domain template.

Pure JAX functions on one-sided frequency-domain arrays. They know nothing
about waveforms, detectors or events: the caller supplies the template and the
data on the band's bins, and the inverse PSD, ``1 / S_n``, on the same bins.
Bins where the PSD is undefined carry an inverse PSD of zero and drop out of
every sum. The last axis is always frequency; leading axes broadcast.

With :math:`\Delta f` the bin spacing, the noise-weighted inner product is

.. math::

    \langle a, b \rangle = 4 \, \Delta f \, \mathrm{Re} \sum_f
    \frac{a^*(f) \, b(f)}{S_n(f)}

and the optimal SNR of a template is :math:`\sigma = \sqrt{\langle h, h \rangle}`.
The complex matched-filter SNR of data :math:`d` against the template is

.. math::

    z(t) = 4 \, \Delta f \sum_f \frac{d(f) \, \hat h^*(f)}{S_n(f)} \,
    e^{2 \pi i f t}, \qquad \hat h = h / \sigma

The template's normalised spectral density :math:`p(f) \propto |h(f)|^2 / S_n(f)`
has the angular-frequency cumulants :math:`\kappa_1` (centroid) and
:math:`\kappa_2` (variance); :math:`\sqrt{\kappa_2}` is the RMS bandwidth that
sets the width :math:`\sim 1 / \sqrt{\kappa_2}` of the SNR peak in time.

The rectangle sum is used throughout, so the template's autocorrelation is
exactly one at zero lag on the grid.

JAX must run in 64-bit mode: call :func:`injstar.runtime.configure_runtime`
first.
"""

import math

import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike
from numpy.typing import NDArray


def band_bins(
    *,
    minimum_frequency: float,
    maximum_frequency: float,
    sampling_frequency: float,
    segment_duration: float,
) -> NDArray[np.int_]:
    """Indices of the band's bins on the periodic segment's frequency grid.

    The segment has ``round(sampling_frequency * segment_duration)`` samples and
    bin ``k`` is at ``k / segment_duration`` Hz. The band runs from the first bin
    at or above ``minimum_frequency`` to the last at or below
    ``maximum_frequency``, kept strictly below the Nyquist frequency.

    Parameters
    ----------
    minimum_frequency, maximum_frequency
        Band edges in Hz.
    sampling_frequency
        Sample rate of the series in Hz.
    segment_duration
        Length of the periodic segment in seconds.

    Returns
    -------
    numpy.ndarray
        Contiguous bin indices; empty if the band holds no bin.
    """
    n_samples = round(sampling_frequency * segment_duration)
    first = math.ceil(minimum_frequency * segment_duration)
    last = min(math.floor(maximum_frequency * segment_duration), n_samples // 2 - 1)
    return np.arange(first, last + 1)


def noise_weighted_inner_product(
    a: ArrayLike, b: ArrayLike, *, inverse_psd: ArrayLike, delta_f: float
) -> jax.Array:
    """Noise-weighted inner product of two frequency-domain series.

    Parameters
    ----------
    a, b
        Frequency-domain series on the band's bins, shape ``(..., bins)``.
    inverse_psd
        ``1 / S_n`` on the same bins, shape ``(..., bins)``; zero where the PSD
        is undefined.
    delta_f
        Frequency spacing of the bins in Hz.

    Returns
    -------
    jax.Array
        Real inner product, shape ``(...)``.
    """
    integrand = jnp.conj(jnp.asarray(a)) * jnp.asarray(b) * inverse_psd
    return 4 * delta_f * jnp.real(jnp.sum(integrand, axis=-1))


def optimal_snr(
    template: ArrayLike, *, inverse_psd: ArrayLike, delta_f: float
) -> jax.Array:
    r"""Optimal matched-filter SNR, :math:`\sqrt{\langle h, h \rangle}`.

    Parameters
    ----------
    template
        Template on the band's bins, shape ``(..., bins)``.
    inverse_psd
        ``1 / S_n`` on the same bins, zero where undefined.
    delta_f
        Frequency spacing of the bins in Hz.

    Returns
    -------
    jax.Array
        Optimal SNR, shape ``(...)``; zero when the template has no weight.
    """
    return jnp.sqrt(
        noise_weighted_inner_product(
            template, template, inverse_psd=inverse_psd, delta_f=delta_f
        )
    )


def first_frequency_cumulant(
    template: ArrayLike, *, frequencies: ArrayLike, inverse_psd: ArrayLike
) -> jax.Array:
    r"""Centroid of the template's angular frequency, :math:`\kappa_1`.

    The mean of :math:`\omega = 2 \pi f` under the density
    :math:`|h|^2 / S_n` normalised to unit sum.

    Parameters
    ----------
    template
        Template on the band's bins, shape ``(..., bins)``.
    frequencies
        Bin frequencies in Hz, shape ``(bins,)``.
    inverse_psd
        ``1 / S_n`` on the same bins, zero where undefined.

    Returns
    -------
    jax.Array
        Centroid in rad/s, shape ``(...)``; zero when the template has no weight.
    """
    omega = 2 * jnp.pi * jnp.asarray(frequencies)
    power = jnp.abs(jnp.asarray(template)) ** 2 * inverse_psd
    total = jnp.sum(power, axis=-1)
    return jnp.sum(omega * power, axis=-1) / jnp.where(total > 0, total, 1.0)


def second_frequency_cumulant(
    template: ArrayLike, *, frequencies: ArrayLike, inverse_psd: ArrayLike
) -> jax.Array:
    r"""Variance of the template's angular frequency, :math:`\kappa_2`.

    The square of the RMS bandwidth: :math:`\langle \omega^2 \rangle -
    \kappa_1^2` under the density :math:`|h|^2 / S_n` normalised to unit sum.

    Parameters
    ----------
    template
        Template on the band's bins, shape ``(..., bins)``.
    frequencies
        Bin frequencies in Hz, shape ``(bins,)``.
    inverse_psd
        ``1 / S_n`` on the same bins, zero where undefined.

    Returns
    -------
    jax.Array
        Variance in (rad/s)², shape ``(...)``, clipped at zero against round-off;
        zero when the template has no weight.
    """
    omega = 2 * jnp.pi * jnp.asarray(frequencies)
    power = jnp.abs(jnp.asarray(template)) ** 2 * inverse_psd
    total = jnp.sum(power, axis=-1)
    safe_total = jnp.where(total > 0, total, 1.0)
    mean = jnp.sum(omega * power, axis=-1) / safe_total
    mean_square = jnp.sum(omega**2 * power, axis=-1) / safe_total
    return jnp.maximum(mean_square - mean**2, 0.0)


def matched_filter_snr_series(
    data: ArrayLike,
    template: ArrayLike,
    *,
    inverse_psd: ArrayLike,
    minimum_frequency: float,
    maximum_frequency: float,
    sampling_frequency: float,
    segment_duration: float,
) -> jax.Array:
    r"""Complex matched-filter SNR time series on a periodic segment.

    Correlates the data with the unit-norm template at every lag with one
    inverse FFT over the segment's frequency grid; see the module docstring
    for the formula. The grid follows from the sampling frequency and the
    segment duration, and the arrays cover the bins of :func:`band_bins`.

    Parameters
    ----------
    data
        Frequency-domain data on the band's bins, shape ``(detectors, bins)``.
    template
        Template on the same bins, shape ``(bins,)`` or ``(detectors, bins)``.
    inverse_psd
        ``1 / S_n`` on the same bins, shape ``(detectors, bins)``; zero where
        undefined.
    minimum_frequency, maximum_frequency
        Band edges in Hz.
    sampling_frequency
        Sample rate of the series in Hz.
    segment_duration
        Length of the periodic segment in seconds.

    Returns
    -------
    jax.Array
        Complex SNR series, shape ``(detectors, n_samples)`` with
        ``n_samples = round(sampling_frequency * segment_duration)``; sample
        ``m`` is at time ``m / sampling_frequency``, periodic in the segment. It
        is zero for a detector whose template has no weight.

    Raises
    ------
    ValueError
        If the arrays do not have one entry per bin of the band.
    """
    bins = band_bins(
        minimum_frequency=minimum_frequency,
        maximum_frequency=maximum_frequency,
        sampling_frequency=sampling_frequency,
        segment_duration=segment_duration,
    )
    n_samples = round(sampling_frequency * segment_duration)
    delta_f = 1.0 / segment_duration
    template = jnp.asarray(template)
    data = jnp.asarray(data, dtype=complex)
    for name, array in (("data", data), ("template", template)):
        if array.shape[-1] != len(bins):
            raise ValueError(
                f"{name} has {array.shape[-1]} bins but the band holds {len(bins)}"
            )
    sigma = optimal_snr(template, inverse_psd=inverse_psd, delta_f=delta_f)
    safe_sigma = jnp.where(sigma > 0, sigma, 1.0)
    spectrum = (
        4
        * delta_f
        * data
        * jnp.conj(template)
        * inverse_psd
        / jnp.expand_dims(safe_sigma, -1)
    )
    full = jnp.zeros((*spectrum.shape[:-1], n_samples), dtype=spectrum.dtype)
    full = full.at[..., bins].set(spectrum)
    return n_samples * jnp.fft.ifft(full, axis=-1)
