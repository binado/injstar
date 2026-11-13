"""BAYESTAR localization.

Packs simulated SNR series, response tensors, locations and horizons into the
arguments of ``ligo.skymap.core.toa_phoa_snr``, runs one call per detected
event, and applies the postprocessing of ``ligo.skymap.bayestar.localize``.

:func:`localize_event` conditions each event as
``ligo.skymap.bayestar.condition`` does: detector positions
and arrival times are centred on their mean weighted by the inverse timing
variance, and the series are trimmed to the light-travel time across the
network plus 5 ms (or a quarter autocorrelation period if that is longer).

Design: "BAYESTAR numerical interface" in docs/design.md; the audited
signature is in docs/audit.md. Nothing here imports JAX.
"""

import os
import time
from dataclasses import dataclass
from pathlib import Path

import lal
import numpy as np
from astropy.table import Table
from ligo.skymap import core, distance, moc
from ligo.skymap.bayestar.filter import unwrap
from ligo.skymap.io import fits
from numpy.typing import NDArray

from injstar import __version__
from injstar.config import PriorConfig
from injstar.constants import ARRIVAL_TIME_MARGIN_S, SPEED_OF_LIGHT_M_S
from injstar.products import skymap_path


@dataclass(frozen=True)
class Event:
    """One simulated event, ready to localize.

    Attributes
    ----------
    event_id
        Identifier from the population.
    coa_time
        Geocentric coalescence GPS time in seconds.
    series
        Complex SNR series, shape ``(detectors, samples)``, ``samples`` odd.
    epochs
        Time of each series' first sample minus ``coa_time``, in seconds.
    snr
        Observed SNR per detector, used for the timing weights.
    bandwidth
        RMS angular-frequency bandwidth per detector in rad/s.
    horizon
        SNR = 1 distance per detector in Mpc.
    """

    event_id: int
    coa_time: float
    series: NDArray[np.complex128]
    epochs: NDArray[np.float64]
    snr: NDArray[np.float64]
    bandwidth: NDArray[np.float64]
    horizon: NDArray[np.float64]


@dataclass(frozen=True)
class Network:
    """Static inputs shared by every event of a run.

    Attributes
    ----------
    names
        Detector names.
    responses
        Earth-fixed response tensors, shape ``(detectors, 3, 3)``.
    locations
        Earth-fixed positions in metres, shape ``(detectors, 3)``.
    sample_rate
        Sample rate of the SNR series in Hz.
    prior
        Distance prior.
    rescale_loglikelihood
        BAYESTAR's log-likelihood scale factor.
    """

    names: tuple[str, ...]
    responses: NDArray[np.float64]
    locations: NDArray[np.float64]
    sample_rate: float
    prior: PriorConfig
    rescale_loglikelihood: float


def localize_event(event: Event, network: Network) -> Table:
    """Localize one event with BAYESTAR.

    Parameters
    ----------
    event
        Simulated event.
    network
        Network geometry, sampling and prior.

    Returns
    -------
    astropy.table.Table
        Multi-order sky map with ``UNIQ``, ``PROBDENSITY``, ``DISTMU``,
        ``DISTSIGMA`` and ``DISTNORM``; ``log_bci`` and ``log_bsn`` are in
        ``meta``.

    Raises
    ------
    ValueError
        If the SNR series has even length or is shorter than BAYESTAR's
        arrival-time window, or if the timing weights are unusable.
    """
    # Condition as ``ligo.skymap.bayestar.condition`` does.
    start = time.perf_counter()
    rate = network.sample_rate
    weights = np.square(event.snr * event.bandwidth)
    if not np.all(np.isfinite(weights)) or weights.sum() <= 0:
        raise ValueError(f"event {event.event_id} has no usable timing weights")

    locations = network.locations / SPEED_OF_LIGHT_M_S
    locations = locations - np.average(locations, weights=weights, axis=0)

    max_abs_t = np.max(np.linalg.norm(locations, axis=1)) + ARRIVAL_TIME_MARGIN_S
    usable = event.bandwidth > 0
    max_abs_t = max(max_abs_t, np.max(0.5 * np.pi / event.bandwidth[usable]))
    desired = 2 * int(np.ceil(max_abs_t * rate)) - 1

    series, epochs = event.series, np.asarray(event.epochs, dtype=float)
    length = series.shape[-1]
    if length % 2 == 0:
        raise ValueError("SNR time series must have odd lengths")
    if length < desired:
        raise ValueError(
            f"crop_duration gives {length} samples but BAYESTAR needs {desired} "
            f"({desired / rate:.4f} s) for this network and band"
        )
    if length > desired:
        trim = (length - desired) // 2
        series = series[:, trim : trim + desired]
        epochs = epochs + trim / rate
        length = desired

    # Arrival times are the centre samples; centre them on their weighted mean.
    toas = epochs + 0.5 * (length - 1) / rate
    mean_toa = np.average(toas, weights=weights)
    seconds = int(np.floor(event.coa_time))
    epoch = lal.LIGOTimeGPS(
        seconds, int(np.round(1e9 * (event.coa_time - seconds + mean_toa)))
    )
    first_samples = toas - mean_toa - 0.5 * (length - 1) / rate

    snrs = np.stack(
        (np.abs(series), np.stack([unwrap(np.angle(row)) for row in series])), axis=-1
    )
    gmst = lal.GreenwichMeanSiderealTime(epoch)

    prior = network.prior
    pixels, log_bci, log_bsn = core.toa_phoa_snr(
        prior.min_distance,
        prior.max_distance,
        prior.distance_power,
        prior.cosmology,
        gmst,
        network.sample_rate,
        first_samples,
        snrs.astype(np.float32),
        np.asarray(network.responses, dtype=np.float32),
        locations,
        np.asarray(event.horizon, dtype=float),
        network.rescale_loglikelihood,
    )
    skymap = Table(pixels, copy=False)

    # As in ligo.skymap.bayestar.localize: moments to ansatz parameters, then
    # the marginal distance moments.
    distmean = skymap.columns.pop("DISTMEAN")
    diststd = skymap.columns.pop("DISTSTD")
    skymap["DISTMU"], skymap["DISTSIGMA"], skymap["DISTNORM"] = (
        distance.moments_to_parameters(distmean, diststd)
    )
    # Pixels whose distance moments BAYESTAR could not evaluate are left out and
    # the rest renormalised; ligo.skymap's own postprocessing does not
    # renormalise, which biases the mean low when the excluded pixels hold
    # probability (see products.map_statistics).
    good = np.isfinite(distmean) & np.isfinite(diststd)
    prob = (moc.uniq2pixarea(skymap["UNIQ"]) * skymap["PROBDENSITY"])[good]
    prob = prob / prob.sum() if prob.sum() > 0 else prob
    rbar = (prob * distmean[good]).sum()
    r2bar = (prob * (np.square(diststd[good]) + np.square(distmean[good]))).sum()

    skymap.meta.update(
        objid=str(event.event_id),
        log_bci=float(log_bci),
        log_bsn=float(log_bsn),
        distmean=float(rbar),
        diststd=float(np.sqrt(r2bar - np.square(rbar))),
        creator="BAYESTAR",
        origin="injstar",
        vcs_version=f"injstar {__version__}",
        gps_time=float(epoch),
        gps_creation_time=lal.GPSTimeNow(),
        instruments=set(network.names),
        runtime=time.perf_counter() - start,
    )
    return skymap


def localize_to_file(event: Event, network: Network, output_dir: Path) -> int:
    """Localize one event and write its sky map; the worker-pool entry point.

    The file is written beside its destination and renamed onto it, so a file
    at the destination is always complete.

    Parameters
    ----------
    event
        Simulated event.
    network
        Network geometry, sampling and prior.
    output_dir
        Run directory.

    Returns
    -------
    int
        The event id.
    """
    skymap = localize_event(event, network)
    target = skymap_path(output_dir, event.event_id)
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(f".{target.name}.{os.getpid()}.partial.fits")
    try:
        partial.unlink(missing_ok=True)
        fits.write_sky_map(str(partial), skymap, moc=True)
        os.replace(partial, target)
    finally:
        partial.unlink(missing_ok=True)
    return event.event_id
