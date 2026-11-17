"""Geodetic detector geometry and response calculations."""

import math
from functools import cached_property
from typing import Annotated

import numpy as np
from numpy.typing import NDArray
from pydantic import ConfigDict, Field
from pydantic.dataclasses import dataclass

from injstar.constants import WGS84_FLATTENING, WGS84_SEMI_MAJOR_AXIS_M


@dataclass(frozen=True, config=ConfigDict(extra="forbid", allow_inf_nan=False))
class DetectorGeometry:
    """Defining geodetic fields; derived arrays are excluded from serialization."""

    latitude_rad: Annotated[float, Field(ge=-math.pi / 2, le=math.pi / 2)]
    longitude_rad: Annotated[float, Field(ge=-math.pi, le=math.pi)]
    elevation_m: Annotated[float, Field(ge=-1e4, le=1e5)]
    xarm_azimuth_rad: float
    yarm_azimuth_rad: float
    xarm_tilt_rad: float = 0.0
    yarm_tilt_rad: float = 0.0

    @cached_property
    def location(self) -> NDArray[np.float64]:
        """Earth-fixed WGS-84 vertex location in metres, read-only."""
        return detector_location(
            latitude_rad=self.latitude_rad,
            longitude_rad=self.longitude_rad,
            elevation_m=self.elevation_m,
        )

    @cached_property
    def response_tensor(self) -> NDArray[np.float64]:
        """Half the difference of arm outer products, read-only float64."""
        return detector_response_tensor(
            latitude_rad=self.latitude_rad,
            longitude_rad=self.longitude_rad,
            xarm_azimuth_rad=self.xarm_azimuth_rad,
            yarm_azimuth_rad=self.yarm_azimuth_rad,
            xarm_tilt_rad=self.xarm_tilt_rad,
            yarm_tilt_rad=self.yarm_tilt_rad,
        )


def detector_location(
    *, latitude_rad: float, longitude_rad: float, elevation_m: float
) -> NDArray[np.float64]:
    """Earth-fixed WGS-84 vertex location in metres, read-only."""
    lat, lon = latitude_rad, longitude_rad
    eccentricity_squared = WGS84_FLATTENING * (2 - WGS84_FLATTENING)
    radius = WGS84_SEMI_MAJOR_AXIS_M / math.sqrt(
        1 - eccentricity_squared * math.sin(lat) ** 2
    )
    result = np.array(
        [
            (radius + elevation_m) * math.cos(lat) * math.cos(lon),
            (radius + elevation_m) * math.cos(lat) * math.sin(lon),
            (radius * (1 - eccentricity_squared) + elevation_m) * math.sin(lat),
        ],
        dtype=np.float64,
    )
    result.setflags(write=False)
    return result


def detector_response_tensor(
    *,
    latitude_rad: float,
    longitude_rad: float,
    xarm_azimuth_rad: float,
    yarm_azimuth_rad: float,
    xarm_tilt_rad: float,
    yarm_tilt_rad: float,
) -> NDArray[np.float64]:
    """Earth-fixed response tensor, the half difference of arm products."""
    lat, lon = latitude_rad, longitude_rad
    north = np.array(
        [
            -math.sin(lat) * math.cos(lon),
            -math.sin(lat) * math.sin(lon),
            math.cos(lat),
        ]
    )
    east = np.array([-math.sin(lon), math.cos(lon), 0.0])
    up = np.array(
        [
            math.cos(lat) * math.cos(lon),
            math.cos(lat) * math.sin(lon),
            math.sin(lat),
        ]
    )

    def arm(azimuth: float, tilt: float) -> NDArray[np.float64]:
        return (
            math.cos(tilt) * (math.cos(azimuth) * north + math.sin(azimuth) * east)
            + math.sin(tilt) * up
        )

    x = arm(xarm_azimuth_rad, xarm_tilt_rad)
    y = arm(yarm_azimuth_rad, yarm_tilt_rad)
    result = (np.outer(x, x) - np.outer(y, y)) / 2
    result.setflags(write=False)
    return result
