"""Read ordered gwmock-pop catalogues for simulation.

Detector-frame masses, distance, sky position, inclination, polarization and
coalescence time are required. Optional spins, tides and phase default to zero.
Extra columns retain their names.
"""

from pathlib import Path

import numpy as np
from gwmock_pop.loaders.file_loader import read_population_catalogue
from numpy.typing import NDArray

#: Columns the pipeline uses, in output order.
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

#: Columns that default to zero when the population lacks them.
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


def read_population(path: Path) -> Population:
    """Read a catalogue in row order, adding defaults.

    Parameters
    ----------
    path
        CSV or HDF5 catalogue in gwmock-pop format.

    Returns
    -------
    dict of str to ndarray
        Simulation columns followed by the remaining input columns.
    """
    columns = read_population_catalogue(path)
    count = len(next(iter(columns.values())))
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
    """The per-event inputs of :class:`injstar.simulate.Simulator`.

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
