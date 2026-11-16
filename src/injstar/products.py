"""Run products.

Computes per-map statistics, stores them in FITS and collects them in CSV.

Nothing here imports JAX.
"""

import csv
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import TYPE_CHECKING, Self

import numpy as np
from numpy.typing import NDArray

from injstar import io

if TYPE_CHECKING:
    from astropy.table import Table

#: Credible levels of the sky areas and distance quantiles in the summary.
CONTOUR_LEVELS = (0.5, 0.9)
DISTANCE_QUANTILES = (0.05, 0.5, 0.95)


@dataclass(frozen=True)
class LocalizationSummary:
    """One localization's statistics and zero-based injection row index.

    Areas are in square degrees, offset in degrees, and distances in Mpc.
    Probabilities and log Bayes factors are dimensionless.
    """

    row_index: int
    area_50: float
    area_90: float
    searched_area: float
    searched_prob: float
    searched_prob_dist: float
    searched_prob_vol: float
    offset: float
    distance_q05: float
    distance_q50: float
    distance_q95: float
    distance_lost_prob: float
    distmean: float
    diststd: float
    log_bci: float
    log_bsn: float

    def dump_fits(self, path: Path) -> None:
        """Append or replace the one-row ``SUMMARY`` extension in a FITS file."""
        from astropy.io import fits

        columns = [fits.Column(name="row_index", format="K", array=[self.row_index])]
        columns.extend(
            fits.Column(name=name, format="D", array=[getattr(self, name)])
            for name in statistic_names()
        )
        summary = fits.BinTableHDU.from_columns(columns, name="SUMMARY")
        with fits.open(path, mode="update") as hdus:
            if "SUMMARY" in hdus:
                hdus["SUMMARY"] = summary
            else:
                hdus.append(summary)

    @classmethod
    def from_fits(cls, path: Path) -> Self:
        """Read stored statistics; missing ``SUMMARY`` raises ``ValueError``."""
        from astropy.io import fits

        with fits.open(path) as hdus:
            if "SUMMARY" not in hdus:
                raise ValueError(f"{path} has no SUMMARY extension")
            row = hdus["SUMMARY"].data[0]
            return cls(
                row_index=int(row["row_index"]),
                **{name: float(row[name]) for name in statistic_names()},
            )


def statistic_names() -> tuple[str, ...]:
    """Names of the per-map columns of the summary, in order."""
    return tuple(
        field.name for field in fields(LocalizationSummary) if field.name != "row_index"
    )


def _distance_statistics(
    prob: NDArray[np.float64],
    distmu: NDArray[np.float64],
    distsigma: NDArray[np.float64],
    distnorm: NDArray[np.float64],
    marginal_ppf: Callable[..., NDArray[np.float64]],
) -> tuple[float, NDArray[np.float64]]:
    """Distance quantiles from the pixels with a valid distance layer.

    Parameters
    ----------
    prob
        Sky probability of each pixel.
    distmu, distsigma, distnorm
        Distance-ansatz parameters of each pixel.
    marginal_ppf
        ``ligo.skymap.distance.marginal_ppf``, passed in to keep that import
        lazy.

    Returns
    -------
    lost_prob
        Fraction of the sky probability in the excluded pixels.
    quantiles
        The :data:`DISTANCE_QUANTILES`; NaN if no pixel is valid.
    """
    valid = np.isfinite(distmu) & np.isfinite(distsigma) & (distnorm > 0)
    total = prob.sum()
    kept = prob[valid].sum()
    if not valid.any() or kept <= 0:
        return 1.0, np.full(len(DISTANCE_QUANTILES), np.nan)
    quantiles = marginal_ppf(
        np.asarray(DISTANCE_QUANTILES),
        prob[valid] / kept,
        distmu[valid],
        distsigma[valid],
        distnorm[valid],
    )
    return float(max(0.0, 1.0 - kept / total)), np.asarray(quantiles)


def map_statistics(
    skymap: "Table",
    *,
    row_index: int,
    right_ascension: float,
    declination: float,
    distance: float,
) -> LocalizationSummary:
    """Statistics of one sky map against the true source position.

    Parameters
    ----------
    skymap
        In-memory multi-order sky map.
    row_index
        Zero-based injection catalogue index.
    right_ascension, declination
        True position in radians.
    distance
        True luminosity distance in Mpc.

    Returns
    -------
    LocalizationSummary
        The columns of :func:`statistic_names`. Areas are in square degrees,
        ``offset`` in degrees, distances in Mpc.

    Notes
    -----
    BAYESTAR can fail to evaluate the distance layer of a pixel: when the
    variance it computes from log-integrals comes out negative it writes
    ``DISTMU = inf``, ``DISTSIGMA = 1`` and ``DISTNORM = 0``. With the PyPI
    macOS arm64 wheel of ligo.skymap 2.5.4, in a zero-noise ET+2CE test (one
    36+29 binary black hole, 16384 Hz), such pixels held 0% of the sky
    probability at network SNR 100, 1% at 300, 8% at 400, 24% at 500 and 52% at
    1000, including the peak pixel. Local builds of the same source did not
    fail, so this is specific to that wheel.

    The distance quantiles, ``distmean`` and ``diststd`` use only the pixels
    with a valid distance layer, with their probabilities renormalised.
    ``distance_lost_prob`` is the sky probability of the excluded pixels, and
    those distance statistics are biased by an unknown amount when it is not
    small: judge each event by it. ``searched_prob_dist`` and
    ``searched_prob_vol`` come from ``ligo.skymap``'s crossmatch, which uses the
    distance layer as it stands, so they are unreliable under the same
    condition. The sky statistics (areas, ``searched_prob``, ``offset``) do not
    depend on the distance layer.
    """
    # Heavy, and only the summary needs them.
    from astropy import units
    from astropy.coordinates import SkyCoord
    from ligo.skymap import distance as skymap_distance
    from ligo.skymap import moc
    from ligo.skymap.postprocess.crossmatch import crossmatch

    truth = SkyCoord(
        right_ascension * units.rad, declination * units.rad, distance * units.Mpc
    )
    match = crossmatch(skymap, truth, contours=CONTOUR_LEVELS)
    prob = moc.uniq2pixarea(skymap["UNIQ"]) * skymap["PROBDENSITY"]
    lost_prob, quantiles = _distance_statistics(
        np.asarray(prob),
        np.asarray(skymap["DISTMU"]),
        np.asarray(skymap["DISTSIGMA"]),
        np.asarray(skymap["DISTNORM"]),
        skymap_distance.marginal_ppf,
    )
    values = (
        *match.contour_areas,
        match.searched_area,
        match.searched_prob,
        match.searched_prob_dist,
        match.searched_prob_vol,
        match.offset,
        *quantiles,
        lost_prob,
        skymap.meta.get("distmean", np.nan),
        skymap.meta.get("diststd", np.nan),
        skymap.meta.get("log_bci", np.nan),
        skymap.meta.get("log_bsn", np.nan),
    )
    return LocalizationSummary(
        row_index=row_index,
        **{
            name: float(value)
            for name, value in zip(statistic_names(), values, strict=True)
        },
    )


def build_summary(output_dir: Path) -> Path:
    """Collect stored FITS summaries into ``summary.csv``, sorted by row index.

    Only completed maps under ``skymaps/`` contribute rows. No pixel data or
    injection catalogue is read. An empty run produces a header-only CSV.
    """
    output_dir = Path(output_dir)
    summaries = sorted(
        (
            LocalizationSummary.from_fits(path)
            for path in (output_dir / "skymaps").glob("*.multiorder.fits")
        ),
        key=lambda summary: summary.row_index,
    )
    target = output_dir / io.SUMMARY_FILE
    with target.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("row_index", *statistic_names()))
        writer.writeheader()
        writer.writerows(asdict(summary) for summary in summaries)
    return target


def collect_summaries(
    directories: Iterable[Path], output: Path, *, overwrite: bool = False
) -> Path:
    """Combine chunk summaries in original row order without moving maps.

    Parameters
    ----------
    directories
        Explicit localization directories from the same population and
        scientific configuration. Provenance is the caller's responsibility.
    output
        Destination CSV. Its parent directory must exist.
    overwrite
        Replace an existing destination.

    Returns
    -------
    pathlib.Path
        The combined summary. Header-only input summaries are accepted.

    Raises
    ------
    ValueError
        If no directories are supplied, columns differ, indices overlap,
        or the destination exists without overwrite permission.
    FileNotFoundError
        If an input summary is missing.
    """
    output = Path(output)
    if output.exists() and not overwrite:
        raise ValueError(f"{output} exists; pass --overwrite to replace it")
    names = ("row_index", *statistic_names())
    rows: dict[int, dict[str, str]] = {}
    count = 0
    for directory in directories:
        count += 1
        path = Path(directory) / io.SUMMARY_FILE
        with path.open(newline="") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames != list(names):
                raise ValueError(f"{path} has unexpected summary columns")
            for row in reader:
                # Parse before opening the destination, including all statistics.
                if None in row or any(row[name] is None for name in names):
                    raise ValueError(f"{path} has an incomplete summary row")
                index = int(row["row_index"])
                if index in rows:
                    raise ValueError(f"duplicate row_index {index}")
                for name in statistic_names():
                    float(row[name])
                rows[index] = row
    if not count:
        raise ValueError("no summary directories supplied")
    with output.open("w" if overwrite else "x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=names)
        writer.writeheader()
        writer.writerows(rows[index] for index in sorted(rows))
    return output
