"""Pure JAX detector projection with the LAL polarization convention.

Coordinates and sidereal angle must refer to compatible celestial frames. No
GPS, epoch, or precession conversion is performed. Angles are radians,
locations metres and delays seconds. Source coordinates and gmst broadcast.
"""

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from injstar.constants import SPEED_OF_LIGHT_M_S


def antenna_pattern(
    response_tensor: ArrayLike,
    gmst: ArrayLike,
    *,
    right_ascension: ArrayLike,
    declination: ArrayLike,
    polarization_angle: ArrayLike,
) -> tuple[jax.Array, jax.Array]:
    """Return plus and cross antenna factors in the gwmock/LAL basis.

    Parameters
    ----------
    response_tensor
        Detector response tensor.
    gmst
        Greenwich mean sidereal time in radians.
    right_ascension
        Source right ascension in radians.
    declination
        Source declination in radians.
    polarization_angle
        Polarization angle in radians.
    """
    gmst, ra, dec, psi = jnp.broadcast_arrays(
        gmst, right_ascension, declination, polarization_angle
    )
    hour = gmst - ra
    ch, sh = jnp.cos(hour), jnp.sin(hour)
    cd, sd = jnp.cos(dec), jnp.sin(dec)
    cp, sp = jnp.cos(psi), jnp.sin(psi)
    x = jnp.stack([-cp * sh - sp * ch * sd, -cp * ch + sp * sh * sd, sp * cd], axis=-1)
    y = jnp.stack([sp * sh - cp * ch * sd, sp * ch + cp * sh * sd, cp * cd], axis=-1)
    tensor = jnp.asarray(response_tensor)
    dx = jnp.einsum("ij,...j->...i", tensor, x)
    dy = jnp.einsum("ij,...j->...i", tensor, y)
    return jnp.sum(x * dx - y * dy, axis=-1), jnp.sum(x * dy + y * dx, axis=-1)


def time_delay_from_geocenter(
    location: ArrayLike,
    gmst: ArrayLike,
    *,
    right_ascension: ArrayLike,
    declination: ArrayLike,
) -> jax.Array:
    """Return minus the source-direction dot location divided by light speed.

    Parameters
    ----------
    location
        Detector location in Earth-fixed coordinates, in metres.
    gmst
        Greenwich mean sidereal time in radians.
    right_ascension
        Source right ascension in radians.
    declination
        Source declination in radians.
    """
    gmst, ra, dec = jnp.broadcast_arrays(gmst, right_ascension, declination)
    hour = gmst - ra
    direction = jnp.stack(
        [jnp.cos(dec) * jnp.cos(hour), -jnp.cos(dec) * jnp.sin(hour), jnp.sin(dec)],
        axis=-1,
    )
    return (
        -jnp.einsum("i,...i->...", jnp.asarray(location), direction)
        / SPEED_OF_LIGHT_M_S
    )
