"""Pure JAX frequency-domain Gaussian detector noise."""

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike


def simulate_fd_noise(
    key: jax.Array, *, inverse_psd: ArrayLike, delta_f: float
) -> jax.Array:
    r"""Draw complex Gaussian noise for an analysis frequency band.

    Each real and imaginary component has variance
    :math:`S_n(f) / (4 \Delta f)`, where :math:`S_n` is the one-sided PSD.
    Bins with nonpositive inverse PSD carry exactly zero noise. This function
    does not impose real-FFT constraints on DC or Nyquist bins or construct a
    complete real-FFT spectrum.

    Parameters
    ----------
    key
        JAX PRNG key. Reusing a key repeats the draw; use ``jax.random.split``
        or ``jax.random.fold_in`` to obtain independent draws. No key is split
        internally and no RNG state is retained.
    inverse_psd
        Finite, nonnegative inverse one-sided PSD on the analysis grid. Its
        shape determines the output shape; leading dimensions can represent
        independent detector spectra.
    delta_f
        Positive, finite frequency spacing in Hz.

    Returns
    -------
    jax.Array
        Complex noise coefficients with the shape of ``inverse_psd``. Random
        draws use JAX's default floating dtype; scaling follows JAX's usual
        dtype promotion. The function supports ``jax.jit`` and ``jax.vmap``.

    Examples
    --------
    Split a key before drawing another independent realization:

    >>> key = jax.random.key(0)
    >>> key, noise_key = jax.random.split(key)
    >>> noise = simulate_fd_noise(noise_key, inverse_psd=jnp.ones(8), delta_f=0.25)
    """
    inverse_psd = jnp.asarray(inverse_psd)
    draw = jax.random.normal(key, (*inverse_psd.shape, 2))
    white = draw[..., 0] + 1j * draw[..., 1]
    psd = jnp.where(
        inverse_psd > 0, 1.0 / jnp.where(inverse_psd > 0, inverse_psd, 1.0), 0.0
    )
    return jnp.sqrt(psd / (4 * delta_f)) * white
