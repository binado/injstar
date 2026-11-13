"""Detector-network specification.

A registry of detectors (geodetic geometry plus a PSD reference) and named
networks, read from TOML. Geometry uses WGS-84, metres and radians, with
azimuths clockwise from North. Arms need not be orthogonal.

The packaged ``data/geometry.toml`` and ``data/sensitivity.toml`` define LIGO,
Virgo, KAGRA, Einstein Telescope and Cosmic Explorer sites. A user's file is
merged on top and uses the same tables::

    [detectors.X1]
    psd_reference = "my_asd.txt"
    psd_kind = "asd"

    [detectors.X1.geometry]
    latitude_rad = 0.76
    longitude_rad = 0.18
    elevation_m = 50.0
    xarm_azimuth_rad = 0.34
    yarm_azimuth_rad = 5.57

    [networks]
    mine = ["X1", "H1", "L1"]

Nothing here imports JAX.
"""

import math
import tomllib
from collections.abc import Mapping, Sequence
from functools import cached_property
from pathlib import Path
from typing import Annotated, Any

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from pydantic.dataclasses import dataclass

from injstar.constants import WGS84_FLATTENING, WGS84_SEMI_MAJOR_AXIS_M
from injstar.detector.psd import PsdKind, Sensitivity, resolve_psd_path

#: The packaged detector geometry and sensitivity tables, the first layers of
#: every merge.
DETECTOR_DEFAULT_PATHS = tuple(
    Path(__file__).parent.parent / "data" / filename
    for filename in ("geometry.toml", "sensitivity.toml")
)

_STRICT = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)


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
        lat, lon = self.latitude_rad, self.longitude_rad
        eccentricity_squared = WGS84_FLATTENING * (2 - WGS84_FLATTENING)
        radius = WGS84_SEMI_MAJOR_AXIS_M / math.sqrt(
            1 - eccentricity_squared * math.sin(lat) ** 2
        )
        result = np.array(
            [
                (radius + self.elevation_m) * math.cos(lat) * math.cos(lon),
                (radius + self.elevation_m) * math.cos(lat) * math.sin(lon),
                (radius * (1 - eccentricity_squared) + self.elevation_m)
                * math.sin(lat),
            ],
            dtype=np.float64,
        )
        result.setflags(write=False)
        return result

    @cached_property
    def response_tensor(self) -> NDArray[np.float64]:
        """Half the difference of arm outer products, read-only float64."""
        lat, lon = self.latitude_rad, self.longitude_rad
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

        x = arm(self.xarm_azimuth_rad, self.xarm_tilt_rad)
        y = arm(self.yarm_azimuth_rad, self.yarm_tilt_rad)
        result = (np.outer(x, x) - np.outer(y, y)) / 2
        result.setflags(write=False)
        return result


class DetectorConfig(BaseModel):
    """Geometry, PSD reference and kind, and an optional presentation label."""

    model_config = _STRICT

    geometry: DetectorGeometry
    psd_reference: Annotated[str, Field(min_length=1, pattern=r"\S")]
    psd_kind: PsdKind = "psd"
    label: Annotated[str, Field(min_length=1)] | None = None

    @model_validator(mode="after")
    def _validate_reference(self) -> "DetectorConfig":
        try:
            resolved = resolve_psd_path(self.psd_reference)
        except FileNotFoundError as error:
            raise ValueError(str(error)) from error
        if isinstance(resolved, Path) and not resolved.is_file():
            raise ValueError(f"PSD reference must be a file: {self.psd_reference!r}")
        return self

    @property
    def sensitivity(self) -> Sensitivity:
        """A fresh :class:`~injstar.detector.psd.Sensitivity` for this detector."""
        return Sensitivity(self.psd_reference, self.psd_kind)


class DetectorRegistry(BaseModel):
    """A complete resolved registry of detectors and named networks."""

    model_config = _STRICT

    detectors: dict[str, DetectorConfig]
    networks: dict[str, tuple[str, ...]] = Field(default_factory=dict)

    @field_validator("detectors")
    @classmethod
    def _validate_names(
        cls, detectors: dict[str, DetectorConfig]
    ) -> dict[str, DetectorConfig]:
        for name in detectors:
            if not name.strip():
                raise ValueError("detector names must be non-empty")
        return detectors

    @model_validator(mode="after")
    def _validate_networks(self) -> "DetectorRegistry":
        """Require each network to be distinct, defined detectors."""
        for name, members in self.networks.items():
            self._check_members(f"network {name!r}", members)
        return self

    def _check_members(self, what: str, members: Sequence[str]) -> None:
        if not members or len(set(members)) != len(members):
            raise ValueError(f"{what} must contain distinct detectors and be non-empty")
        missing = [member for member in members if member not in self.detectors]
        if missing:
            raise ValueError(f"{what} contains undefined detectors: {missing}")

    def members(self, network: str | Sequence[str]) -> tuple[str, ...]:
        """Detector names of a network, in declared order.

        Parameters
        ----------
        network
            A name from ``networks``, or a sequence of detector names.

        Raises
        ------
        ValueError
            If the name is unknown or a listed detector is undefined.
        """
        if isinstance(network, str):
            if network not in self.networks:
                raise ValueError(
                    f"unknown network {network!r}; defined: {sorted(self.networks)}"
                )
            return self.networks[network]
        members = tuple(network)
        self._check_members("network", members)
        return members

    def select(self, network: str | Sequence[str]) -> "DetectorRegistry":
        """The registry holding only ``network`` and the detectors it uses.

        Parameters
        ----------
        network
            A name from ``networks``, or a sequence of detector names.
        """
        members = self.members(network)
        return DetectorRegistry(
            detectors={name: self.detectors[name] for name in members},
            networks={"selected": members},
        )

    def arrays(
        self, network: str | Sequence[str]
    ) -> tuple[tuple[str, ...], NDArray[np.float64], NDArray[np.float64]]:
        """Stacked geometry of a network.

        Parameters
        ----------
        network
            A name from ``networks``, or a sequence of detector names.

        Returns
        -------
        names : tuple of str
            Detector names, in declared order.
        responses : ndarray, shape (n, 3, 3)
            Earth-fixed response tensors.
        locations : ndarray, shape (n, 3)
            Earth-fixed vertex positions in metres.
        """
        names = self.members(network)
        geometries = [self.detectors[name].geometry for name in names]
        responses = np.stack([geometry.response_tensor for geometry in geometries])
        locations = np.stack([geometry.location for geometry in geometries])
        return names, responses, locations


def _deep_merge(base: dict[str, Any], layer: Mapping[str, Any]) -> dict[str, Any]:
    """Merge ``layer`` into ``base``: tables recurse, everything else replaces."""
    for key, value in layer.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), dict):
            _deep_merge(base[key], value)
        else:
            base[key] = dict(value) if isinstance(value, Mapping) else value
    return base


def _anchor_psd_references(layer: dict[str, Any], directory: Path) -> None:
    """Make PSD references that are files beside the layer absolute."""
    detectors = layer.get("detectors", {})
    if not isinstance(detectors, dict):
        return
    for table in detectors.values():
        reference = table.get("psd_reference") if isinstance(table, dict) else None
        if isinstance(reference, str) and (directory / reference).is_file():
            table["psd_reference"] = str((directory / reference).resolve())


def load_detector_config(paths: Sequence[Path] = ()) -> DetectorRegistry:
    """Merge packaged detector definitions and ordered TOML registry layers.

    Later layers override earlier geometry, sensitivities and network
    membership. A relative ``psd_reference`` naming a file beside its layer is
    resolved against that layer's directory. Only ``[detectors]`` and
    ``[networks]`` are accepted. Validation checks that each PSD reference
    resolves to a file; the curve itself is read on first use.

    Parameters
    ----------
    paths
        Existing TOML files, merged in order after the packaged defaults.

    Raises
    ------
    ValueError
        If the merged config is not a valid registry.
    """
    merged: dict[str, Any] = {}
    for path in (*DETECTOR_DEFAULT_PATHS, *paths):
        with Path(path).open("rb") as handle:
            layer = tomllib.load(handle)
        _anchor_psd_references(layer, Path(path).resolve().parent)
        _deep_merge(merged, layer)
    return DetectorRegistry.model_validate(merged)
