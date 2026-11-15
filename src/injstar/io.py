"""Run file names and catalogue output."""

from collections.abc import Mapping
from pathlib import Path

import numpy as np
from gwmock_pop.loaders.file_loader import write_population_catalogue
from numpy.typing import NDArray

SNR_FILE = "snr.csv"
INJECTIONS_FILE = "injections.h5"
SUMMARY_FILE = "summary.h5"
CONFIG_FILE = "config.json"


def write_table(path: Path, columns: Mapping[str, NDArray]) -> Path:
    """Write a gwmock-pop HDF5 catalogue, encoding text as UTF-8 bytes."""
    arrays = {
        name: np.char.encode(values, "utf-8") if values.dtype.kind == "U" else values
        for name, values in columns.items()
    }
    write_population_catalogue(path, arrays)
    return path


def skymap_path(output_dir: Path, row_index: int) -> Path:
    """Where the sky map of ``row_index`` lives in a run directory.

    Parameters
    ----------
    output_dir
        Run directory.
    row_index
        Zero-based input row index.
    """
    return Path(output_dir) / "skymaps" / f"{int(row_index)}.multiorder.fits"
