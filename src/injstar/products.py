"""Run products.

Reads and writes the tables of a run directory (``injections.h5`` and
``summary.h5``, one row per injection, non-detections included), the run
manifest, and the per-map statistics that make up the summary.

A table is an HDF5 file with one dataset per column, written beside its
destination and renamed onto it. The atomic write is ported from
``astrogwb.utils.io`` (astrogwb 0.7.0).

Design: "Outputs" in docs/design.md. Nothing here imports JAX.
"""

import hashlib
import json
import os
import tempfile
from collections.abc import Generator, Mapping
from contextlib import contextmanager
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from numpy.typing import NDArray

from injstar import __version__

INJECTIONS_FILE = "injections.h5"
SUMMARY_FILE = "summary.h5"
MANIFEST_FILE = "manifest.json"
CONFIG_FILE = "config.toml"
DETECTORS_FILE = "detectors.toml"

#: Distributions whose versions the manifest records.
RECORDED_DEPENDENCIES = (
    "jax",
    "ripplegw",
    "ligo.skymap",
    "lalsuite",
    "astropy",
    "numpy",
    "gwmock-noise",
)

#: Credible levels of the sky areas and distance quantiles in the summary.
CONTOUR_LEVELS = (0.5, 0.9)
DISTANCE_QUANTILES = (0.05, 0.5, 0.95)

Columns = dict[str, NDArray]


def file_sha256(path: Path) -> str:
    """Hex SHA-256 of a file's contents.

    Parameters
    ----------
    path
        File to hash.
    """
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def fingerprint(record: Mapping[str, Any]) -> str:
    """Hex SHA-256 of a JSON-serializable record, independent of key order.

    Parameters
    ----------
    record
        What defines a run's results.
    """
    encoded = json.dumps(record, sort_keys=True, default=str).encode()
    return hashlib.sha256(encoded).hexdigest()


@contextmanager
def atomic_h5(path: Path) -> Generator[h5py.File]:
    """An ``h5py.File`` open for writing beside ``path``, renamed onto it on success.

    A reader sees no file at ``path`` or a complete one: if the body raises,
    the temporary file is removed and ``path`` is left as it was.

    Parameters
    ----------
    path
        File path.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(
        dir=target.parent, prefix=f".{target.stem}.", suffix=".h5.tmp"
    )
    os.close(handle)
    try:
        with h5py.File(temporary, "w") as file:
            yield file
        os.replace(temporary, target)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def write_table(path: Path, columns: Mapping[str, NDArray], **attrs: str) -> Path:
    """Write equal-length columns as one HDF5 dataset each.

    Parameters
    ----------
    path
        Destination file.
    columns
        Column name to 1-D array. Text columns are stored as UTF-8 bytes.
    **attrs
        String attributes recorded on the file, beside ``version`` and
        ``created``.

    Raises
    ------
    ValueError
        If the columns differ in length.
    """
    arrays = {name: np.asarray(values) for name, values in columns.items()}
    if len({len(values) for values in arrays.values()}) > 1:
        raise ValueError("table columns have different lengths")
    with atomic_h5(path) as file:
        file.attrs["version"] = __version__
        file.attrs["created"] = datetime.now(UTC).isoformat()
        for name, value in attrs.items():
            file.attrs[name] = value
        file.attrs["columns"] = json.dumps(list(arrays))
        for name, values in arrays.items():
            if values.dtype.kind == "U":
                values = np.char.encode(values, "utf-8")
            file.create_dataset(name, data=values)
    return Path(path)


def read_table(path: Path) -> tuple[Columns, dict[str, str]]:
    """Read a file :func:`write_table` made.

    Parameters
    ----------
    path
        Table file.

    Returns
    -------
    columns : dict of str to ndarray
        Columns in the order they were written; text columns as ``str``.
    attrs : dict of str to str
        File attributes.
    """
    with h5py.File(path, "r") as file:
        attrs = {
            name: value.decode() if isinstance(value, bytes) else str(value)
            for name, value in file.attrs.items()
        }
        columns: Columns = {}
        for name in json.loads(attrs.pop("columns")):
            values = np.asarray(file[name][()])
            if values.dtype.kind == "S":
                values = np.char.decode(values, "utf-8")
            columns[name] = values
    return columns, attrs


def write_manifest(output_dir: Path, record: Mapping[str, Any]) -> Path:
    """Write ``manifest.json`` with ``record`` and the software versions.

    Parameters
    ----------
    output_dir
        Run directory.
    record
        Inputs of the run; its :func:`fingerprint` is stored alongside.
    """
    manifest = {
        "fingerprint": fingerprint(record),
        "created": datetime.now(UTC).isoformat(),
        "versions": {
            "injstar": __version__,
            **{name: version(name) for name in RECORDED_DEPENDENCIES},
        },
        **record,
    }
    target = Path(output_dir) / MANIFEST_FILE
    partial = target.with_name(f".{target.name}.partial")
    partial.write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    os.replace(partial, target)
    return target


def read_manifest(output_dir: Path) -> dict[str, Any] | None:
    """The manifest of a run directory, or ``None`` if it has none.

    Parameters
    ----------
    output_dir
        Run directory.
    """
    path = Path(output_dir) / MANIFEST_FILE
    return json.loads(path.read_text()) if path.is_file() else None


def skymap_path(output_dir: Path, event_id: int) -> Path:
    """Where the sky map of ``event_id`` lives in a run directory.

    Parameters
    ----------
    output_dir
        Run directory.
    event_id
        Identifier from the population.
    """
    return Path(output_dir) / "skymaps" / f"{int(event_id)}.multiorder.fits"


def statistic_names() -> tuple[str, ...]:
    """Names of the per-map columns of the summary, in order."""
    return (
        *(f"area_{round(100 * level)}" for level in CONTOUR_LEVELS),
        "searched_area",
        "searched_prob",
        "searched_prob_dist",
        "searched_prob_vol",
        "offset",
        *(f"distance_q{round(100 * q):02d}" for q in DISTANCE_QUANTILES),
        "distmean",
        "diststd",
        "log_bci",
        "log_bsn",
    )


def map_statistics(
    path: Path, *, right_ascension: float, declination: float, distance: float
) -> dict[str, float]:
    """Statistics of one sky map against the true source position.

    Parameters
    ----------
    path
        Multi-order FITS sky map.
    right_ascension, declination
        True position in radians.
    distance
        True luminosity distance in Mpc.

    Returns
    -------
    dict of str to float
        The columns of :func:`statistic_names`. Areas are in square degrees,
        ``offset`` in degrees, distances in Mpc.
    """
    # Heavy, and only the summary needs them.
    from astropy import units
    from astropy.coordinates import SkyCoord
    from ligo.skymap import distance as skymap_distance
    from ligo.skymap import moc
    from ligo.skymap.io import read_sky_map
    from ligo.skymap.postprocess.crossmatch import crossmatch

    skymap = read_sky_map(str(path), moc=True)
    truth = SkyCoord(
        right_ascension * units.rad, declination * units.rad, distance * units.Mpc
    )
    match = crossmatch(skymap, truth, contours=CONTOUR_LEVELS)
    prob = moc.uniq2pixarea(skymap["UNIQ"]) * skymap["PROBDENSITY"]
    quantiles = skymap_distance.marginal_ppf(
        np.asarray(DISTANCE_QUANTILES),
        np.asarray(prob),
        np.asarray(skymap["DISTMU"]),
        np.asarray(skymap["DISTSIGMA"]),
        np.asarray(skymap["DISTNORM"]),
    )
    values = (
        *match.contour_areas,
        match.searched_area,
        match.searched_prob,
        match.searched_prob_dist,
        match.searched_prob_vol,
        match.offset,
        *quantiles,
        skymap.meta.get("distmean", np.nan),
        skymap.meta.get("diststd", np.nan),
        skymap.meta.get("log_bci", np.nan),
        skymap.meta.get("log_bsn", np.nan),
    )
    return {
        name: float(value)
        for name, value in zip(statistic_names(), values, strict=True)
    }


def build_summary(output_dir: Path) -> Path:
    """Join ``injections.h5`` with the statistics of every sky map present.

    Rows without a sky map (non-detections, or events not localized yet) get
    NaN statistics and an empty ``skymap`` path.

    Parameters
    ----------
    output_dir
        Run directory.

    Returns
    -------
    pathlib.Path
        The written ``summary.h5``.
    """
    output_dir = Path(output_dir)
    columns, attrs = read_table(output_dir / INJECTIONS_FILE)
    count = len(columns["event_id"])
    statistics = {name: np.full(count, np.nan) for name in statistic_names()}
    paths = np.full(count, "", dtype=object)
    for row, event_id in enumerate(columns["event_id"]):
        path = skymap_path(output_dir, int(event_id))
        if not path.is_file():
            continue
        values = map_statistics(
            path,
            right_ascension=float(columns["right_ascension"][row]),
            declination=float(columns["declination"][row]),
            distance=float(columns["distance"][row]),
        )
        for name, value in values.items():
            statistics[name][row] = value
        paths[row] = str(path.relative_to(output_dir))
    columns["localized"] = paths != ""
    columns.update(statistics)
    columns["skymap"] = paths.astype(str)
    return write_table(
        output_dir / SUMMARY_FILE, columns, fingerprint=attrs.get("fingerprint", "")
    )
