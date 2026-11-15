"""Pure JAX detector projection with the LAL polarization convention.

Coordinates and sidereal angle must refer to compatible celestial frames. No
GPS, epoch, or precession conversion is performed. Angles are radians,
locations metres and delays seconds. Source coordinates and gmst broadcast.
"""

from dataclasses import dataclass
from typing import Protocol

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from injstar.constants import SPEED_OF_LIGHT_M_S
from injstar.detector.network import DetectorGeometry


class Detector[ArrayT](Protocol):
    """Geometry and sky projection for one configured detector.

    All angles are in radians. ``gmst`` is the Greenwich mean sidereal angle,
    not a GPS time, and it broadcasts against the sky coordinates.
    """

    def response_tensor(self) -> ArrayT:
        """Return the Earth-fixed response tensor.

        Returns
        -------
        ArrayT
            Array of shape ``(3, 3)``.
        """
        ...

    def location(self) -> ArrayT:
        """Return the Earth-fixed detector vertex.

        Returns
        -------
        ArrayT
            Array of shape ``(3,)`` in metres.
        """
        ...

    def antenna_pattern(
        self,
        gmst: ArrayT | float,
        *,
        right_ascension: ArrayT | float,
        declination: ArrayT | float,
        polarization_angle: ArrayT | float,
    ) -> tuple[ArrayT, ArrayT]:
        """Return the plus and cross antenna factors.

        The polarization basis follows the LAL polarization-angle convention.

        Parameters
        ----------
        gmst
            Greenwich mean sidereal angle in radians.
        right_ascension
            Source right ascension in radians.
        declination
            Source declination in radians.
        polarization_angle
            Polarization angle in radians.

        Returns
        -------
        plus, cross : ArrayT
            Antenna factors with the broadcast shape of the inputs.
        """
        ...

    def time_delay_from_geocenter(
        self,
        gmst: ArrayT | float,
        *,
        right_ascension: ArrayT | float,
        declination: ArrayT | float,
    ) -> ArrayT:
        """Return detector arrival minus geocentric arrival.

        Parameters
        ----------
        gmst
            Greenwich mean sidereal angle in radians.
        right_ascension
            Source right ascension in radians.
        declination
            Source declination in radians.

        Returns
        -------
        ArrayT
            Delay in seconds with the broadcast shape of the inputs.
        """
        ...


@dataclass(frozen=True)
class GroundDetector:
    """Projection for one ground-based detector with fixed geodetic geometry.

    Implements :class:`Detector` with JAX arrays on top of the module-level
    :func:`antenna_pattern` and :func:`time_delay_from_geocenter`. The response
    tensor comes from the geodetic geometry, so it differs slightly from the
    one LAL tabulates for the same detector.

    Attributes
    ----------
    geometry : DetectorGeometry
        Geodetic location and arm orientation of the detector.
    """

    geometry: DetectorGeometry

    def response_tensor(self) -> jax.Array:
        """Return the Earth-fixed response tensor of shape ``(3, 3)``."""
        return jnp.asarray(self.geometry.response_tensor)

    def location(self) -> jax.Array:
        """Return the Earth-fixed vertex of shape ``(3,)`` in metres."""
        return jnp.asarray(self.geometry.location)

    def antenna_pattern(
        self,
        gmst: ArrayLike,
        *,
        right_ascension: ArrayLike,
        declination: ArrayLike,
        polarization_angle: ArrayLike,
    ) -> tuple[jax.Array, jax.Array]:
        """Return the plus and cross antenna factors.

        See :func:`antenna_pattern` for the parameters and conventions.
        """
        return antenna_pattern(
            self.response_tensor(),
            gmst,
            right_ascension=right_ascension,
            declination=declination,
            polarization_angle=polarization_angle,
        )

    def time_delay_from_geocenter(
        self,
        gmst: ArrayLike,
        *,
        right_ascension: ArrayLike,
        declination: ArrayLike,
    ) -> jax.Array:
        """Return detector arrival minus geocentric arrival in seconds.

        See :func:`time_delay_from_geocenter` for the parameters.
        """
        return time_delay_from_geocenter(
            self.location(),
            gmst,
            right_ascension=right_ascension,
            declination=declination,
        )


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
