"""Frequency-domain noise normalization and JAX transformation contracts."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from injstar.detector.noise import simulate_fd_noise

from ..tolerances import assert_close


@pytest.mark.parametrize("shape", [(), (8,), (2, 8), (0,)])
def test_shape_and_complex_output(shape: tuple[int, ...]) -> None:
    noise = simulate_fd_noise(
        jax.random.key(0), inverse_psd=np.ones(shape), delta_f=0.25
    )

    assert noise.shape == shape
    assert jnp.iscomplexobj(noise)


def test_nonpositive_weights_have_zero_noise() -> None:
    noise = simulate_fd_noise(
        jax.random.key(0), inverse_psd=np.array([0.0, -1.0, 2.0]), delta_f=0.25
    )

    np.testing.assert_array_equal(noise[:2], np.zeros(2, dtype=complex))
    assert np.isfinite(noise).all()
    assert noise[2] != 0


@pytest.mark.parametrize("legacy_key", [False, True])
def test_explicit_key_semantics(legacy_key: bool) -> None:
    key = jax.random.PRNGKey(42) if legacy_key else jax.random.key(42)
    first_key, second_key = jax.random.split(key)
    kwargs = {"inverse_psd": np.ones(32), "delta_f": 0.25}
    first = simulate_fd_noise(first_key, **kwargs)

    np.testing.assert_array_equal(first, simulate_fd_noise(first_key, **kwargs))
    assert not np.array_equal(first, simulate_fd_noise(second_key, **kwargs))


def test_jit_and_vmap_match_eager() -> None:
    keys = jax.random.split(jax.random.key(42), 4)
    inverse_psd = jnp.array([[0.0, 1.0, 2.0], [4.0, 0.0, 8.0]])

    def draw(key: jax.Array) -> jax.Array:
        return simulate_fd_noise(key, inverse_psd=inverse_psd, delta_f=0.25)

    expected = jnp.stack([draw(key) for key in keys])
    assert_close(jax.vmap(draw)(keys), expected)
    assert_close(jax.jit(jax.vmap(draw))(keys), expected)
    assert_close(
        jax.jit(simulate_fd_noise)(keys[0], inverse_psd=inverse_psd, delta_f=0.25),
        expected[0],
    )


@pytest.mark.parametrize("delta_f", [0.25, 2.0])
def test_variance_per_quadrature(delta_f: float) -> None:
    n_draws = 32768
    psd = np.array([0.5, 2.0, 8.0])
    noise = np.asarray(
        simulate_fd_noise(
            jax.random.key(42),
            inverse_psd=np.broadcast_to(1 / psd, (n_draws, len(psd))),
            delta_f=delta_f,
        )
    )
    expected_variance = psd / (4 * delta_f)
    # A Gaussian sample variance has relative standard error sqrt(2 / (n - 1)).
    relative_error = 6 * np.sqrt(2 / (n_draws - 1))
    for quadrature in (noise.real, noise.imag):
        np.testing.assert_allclose(
            quadrature.var(axis=0, ddof=1), expected_variance, rtol=relative_error
        )
        assert np.all(
            np.abs(quadrature.mean(axis=0)) < 6 * np.sqrt(expected_variance / n_draws)
        )
