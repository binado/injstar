"""Injection populations.

Reads a named-column population catalogue in the layout gwmock-pop writes: a
header-based numeric CSV, or an HDF5 file with a structured dataset named
``data`` (a group of equal-length 1-D datasets is accepted too). Column names
follow gwmock-pop's canonical CBC parameter names.

Required per event: ``detector_frame_mass_1``, ``detector_frame_mass_2`` (or
``source_frame_mass_1``, ``source_frame_mass_2`` with ``redshift``),
``distance`` in Mpc (alias ``luminosity_distance``), ``right_ascension``,
``declination``, ``inclination``, ``polarization_angle`` and ``coa_time`` (GPS
seconds at the geocentre). ``spin_1z`` and ``spin_2z`` default to zero.
``event_id`` defaults to the row index. Masses are in solar masses and angles
in radians.

No SNR or detection column is read: every other column is carried through
unchanged under a ``population_`` prefix.

Design: "Inputs" in docs/design.md. Nothing here imports JAX.
"""

from pathlib import Path

import h5py
import numpy as np
from numpy.typing import NDArray

#: Columns the pipeline uses, in output order.
CANONICAL_COLUMNS = (
    "event_id",
    "detector_frame_mass_1",
    "detector_frame_mass_2",
    "spin_1z",
    "spin_2z",
    "distance",
    "right_ascension",
    "declination",
    "inclination",
    "polarization_angle",
    "coa_time",
)

_ALIASES = {"luminosity_distance": "distance"}
_HDF5_DATASET_NAME = "data"

Population = dict[str, NDArray]


def _read_csv(path: Path) -> Population:
    table = np.atleast_1d(
        np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding="utf-8")
    )
    return {name: np.asarray(table[name]) for name in table.dtype.names or ()}


def _read_hdf5(path: Path) -> Population:
    with h5py.File(path, "r") as file:
        if _HDF5_DATASET_NAME not in file:
            raise ValueError(f"{path} has no {_HDF5_DATASET_NAME!r} dataset or group")
        node = file[_HDF5_DATASET_NAME]
        if isinstance(node, h5py.Group):
            return {name: np.asarray(item[()]) for name, item in node.items()}
        table = np.atleast_1d(node[()])
        if table.dtype.names is None:
            raise ValueError(f"{path}: dataset 'data' has no named columns")
        return {name: np.asarray(table[name]) for name in table.dtype.names}


def _detector_frame_masses(columns: Population, cosmology: str) -> None:
    """Fill detector-frame masses in place from source-frame ones if needed."""
    detector = [f"detector_frame_mass_{i}" in columns for i in (1, 2)]
    source = [f"source_frame_mass_{i}" in columns for i in (1, 2)]
    if all(detector) and any(source):
        raise ValueError(
            "population gives both detector-frame and source-frame masses; keep one"
        )
    if all(detector):
        return
    if not all(source):
        raise ValueError(
            "population needs detector_frame_mass_1 and detector_frame_mass_2, "
            "or source_frame_mass_1 and source_frame_mass_2 with redshift"
        )
    if "redshift" not in columns:
        raise ValueError("source-frame masses need a redshift column")
    redshift = np.asarray(columns["redshift"], dtype=float)
    for i in (1, 2):
        columns[f"detector_frame_mass_{i}"] = np.asarray(
            columns.pop(f"source_frame_mass_{i}"), dtype=float
        ) * (1 + redshift)
    if "distance" not in columns:
        # Heavy and only needed on this branch.
        import astropy.cosmology

        model = getattr(astropy.cosmology, cosmology, None)
        if not isinstance(model, astropy.cosmology.FLRW):
            raise ValueError(f"unknown astropy cosmology {cosmology!r}")
        columns["distance"] = model.luminosity_distance(redshift).to_value("Mpc")


def read_population(path: Path, *, cosmology: str = "Planck18") -> Population:
    """Read a population catalogue into canonical columns.

    Parameters
    ----------
    path
        ``.csv``, ``.h5`` or ``.hdf5`` file.
    cosmology
        Name of the astropy cosmology that turns redshift into luminosity
        distance when the file gives source-frame masses and no distance.

    Returns
    -------
    dict of str to ndarray
        The :data:`CANONICAL_COLUMNS` as float64 (``event_id`` as uint64),
        followed by every other column of the file under a ``population_``
        prefix.

    Raises
    ------
    ValueError
        If the format is unsupported, a required column is missing, event ids
        repeat, or a value is not finite.
    """
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        columns = _read_csv(path)
    elif suffix in {".h5", ".hdf5"}:
        columns = _read_hdf5(path)
    else:
        raise ValueError(f"unsupported population format {suffix!r}; use CSV or HDF5")

    for alias, name in _ALIASES.items():
        if alias in columns:
            if name in columns:
                raise ValueError(f"population has both {alias!r} and {name!r}")
            columns[name] = columns.pop(alias)
    _detector_frame_masses(columns, cosmology)

    sizes = {len(np.atleast_1d(column)) for column in columns.values()}
    if len(sizes) != 1:
        raise ValueError("population columns have different lengths")
    (count,) = sizes

    columns.setdefault("event_id", np.arange(count))
    columns.setdefault("spin_1z", np.zeros(count))
    columns.setdefault("spin_2z", np.zeros(count))
    missing = [name for name in CANONICAL_COLUMNS if name not in columns]
    if missing:
        raise ValueError(f"population lacks the columns {missing}")

    event_id = np.asarray(columns["event_id"])
    if np.any(event_id != np.round(event_id)) or np.any(event_id < 0):
        raise ValueError("event_id must hold non-negative integers")
    event_id = event_id.astype(np.uint64)
    if len(np.unique(event_id)) != count:
        raise ValueError("event_id values must be unique")

    out: Population = {"event_id": event_id}
    for name in CANONICAL_COLUMNS[1:]:
        values = np.asarray(columns[name], dtype=float)
        if not np.all(np.isfinite(values)):
            raise ValueError(f"population column {name!r} has non-finite values")
        out[name] = values
    for name in ("detector_frame_mass_1", "detector_frame_mass_2", "distance"):
        if np.any(out[name] <= 0):
            raise ValueError(f"population column {name!r} must be positive")
    for name, values in columns.items():
        if name not in CANONICAL_COLUMNS:
            out[f"population_{name}"] = np.asarray(values)
    return out


def simulation_events(population: Population, gmst: NDArray) -> Population:
    """The per-event inputs of :class:`injstar.simulate.Simulator`.

    Parameters
    ----------
    population
        Output of :func:`read_population`.
    gmst
        Greenwich mean sidereal time in radians at each ``coa_time``.
    """
    mass_1 = population["detector_frame_mass_1"]
    mass_2 = population["detector_frame_mass_2"]
    total = mass_1 + mass_2
    return {
        "event_id": population["event_id"],
        "chirp_mass": (mass_1 * mass_2) ** 0.6 / total**0.2,
        "symmetric_mass_ratio": mass_1 * mass_2 / total**2,
        "spin_1z": population["spin_1z"],
        "spin_2z": population["spin_2z"],
        "luminosity_distance": population["distance"],
        "right_ascension": population["right_ascension"],
        "declination": population["declination"],
        "polarization_angle": population["polarization_angle"],
        "inclination": population["inclination"],
        "gmst": np.asarray(gmst, dtype=float),
    }
