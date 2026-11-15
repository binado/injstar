"""Read and write run files."""

import json
import os
import tempfile
from collections.abc import Generator, Mapping
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

import h5py
import numpy as np
from numpy.typing import NDArray

from injstar import __version__

INJECTIONS_FILE = "injections.h5"
SUMMARY_FILE = "summary.h5"
CONFIG_FILE = "config.json"

Columns = dict[str, NDArray]


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


def skymap_path(output_dir: Path, event_id: int) -> Path:
    """Where the sky map of ``event_id`` lives in a run directory.

    Parameters
    ----------
    output_dir
        Run directory.
    event_id
        Zero-based input row index.
    """
    return Path(output_dir) / "skymaps" / f"{int(event_id)}.multiorder.fits"
