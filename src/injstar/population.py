"""Read ordered gwmock-pop catalogues for simulation.

Detector-frame masses, distance, sky position, inclination, polarization and
coalescence time are required. Optional spins, tides and phase default to zero.
Extra columns retain their names.
"""

from pathlib import Path

import h5py
import numpy as np
import polars as pl
from gwmock_pop.loaders.file_loader import infer_population_file_format
from numpy.typing import NDArray

CANONICAL_COLUMNS = (
    "detector_frame_mass_1",
    "detector_frame_mass_2",
    "spin_1x",
    "spin_1y",
    "spin_1z",
    "spin_2x",
    "spin_2y",
    "spin_2z",
    "lambda_1",
    "lambda_2",
    "distance",
    "right_ascension",
    "declination",
    "inclination",
    "coa_phase",
    "polarization_angle",
    "coa_time",
)

_ZERO_DEFAULTS = (
    "spin_1x",
    "spin_1y",
    "spin_1z",
    "spin_2x",
    "spin_2y",
    "spin_2z",
    "lambda_1",
    "lambda_2",
    "coa_phase",
)

Population = dict[str, NDArray]


def read_population(
    path: Path, *, start: int = 0, stop: int | None = None
) -> Population:
    """Read a catalogue in row order, adding defaults.

    Parameters
    ----------
    path
        CSV or HDF5 catalogue in gwmock-pop format.
    start, stop
        Zero-based, half-open row range. Omitted stop reads to EOF; stops
        beyond EOF are clipped. CSV scans preceding lines sequentially;
        prefer HDF5 for large populations.

    Returns
    -------
    dict of str to ndarray
        Simulation columns followed by the remaining input columns.
    """
    if start < 0 or (stop is not None and (stop < 0 or stop <= start)):
        raise ValueError("population range requires 0 <= start < stop")
    if infer_population_file_format(path) == "csv":
        length = None if stop is None else stop - start
        frame = pl.scan_csv(
            path,
            skip_rows_after_header=start,
            n_rows=length,
            infer_schema_length=length,
            raise_if_empty=False,
        ).collect()
        columns = {
            name: series.to_numpy().astype(str)
            if series.dtype == pl.String
            else series.to_numpy()
            for name, series in zip(frame.columns, frame, strict=True)
        }
    else:
        with h5py.File(path, "r") as file:
            source = file["data"]
            if isinstance(source, h5py.Dataset):
                data = source[start:stop]
                if data.dtype.names is None:
                    raise ValueError("HDF5 catalogue must provide named columns")
                columns = {name: data[name] for name in data.dtype.names}
            else:
                columns = {name: source[name][start:stop] for name in source}
    count = len(next(iter(columns.values()))) if columns else 0
    if count == 0:
        raise ValueError("population selection contains no rows")
    for name in _ZERO_DEFAULTS:
        columns.setdefault(name, np.zeros(count))
    missing = [name for name in CANONICAL_COLUMNS if name not in columns]
    if missing:
        raise ValueError(f"population lacks the columns {missing}")

    out: Population = {}
    for name in CANONICAL_COLUMNS:
        values = np.asarray(columns[name], dtype=float)
        if not np.all(np.isfinite(values)):
            raise ValueError(f"population column {name!r} has non-finite values")
        out[name] = values
    for name in ("detector_frame_mass_1", "detector_frame_mass_2", "distance"):
        if np.any(out[name] <= 0):
            raise ValueError(f"population column {name!r} must be positive")
    for name, values in columns.items():
        if name not in CANONICAL_COLUMNS:
            out[name] = np.asarray(values)
    return out


def simulation_events(population: Population, gmst: NDArray) -> Population:
    """The per-event inputs of :func:`injstar.inject.inject_events`.

    Parameters
    ----------
    population
        Output of :func:`read_population`.
    gmst
        Greenwich mean sidereal time in radians at each ``coa_time``.
    """
    events = {
        name: population[name] for name in CANONICAL_COLUMNS if name != "coa_time"
    }
    events["gmst"] = np.asarray(gmst, dtype=float)
    return events
