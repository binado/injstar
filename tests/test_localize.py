from pathlib import Path

import lal
import numpy as np
import pytest
from ligo.skymap.bayestar import log_posterior_toa_phoa_snr

from injstar.localize import Event, Network, condition_event, localize_event
from injstar.products import map_statistics


def test_localize_event_rejects_a_short_series(
    event_and_network: tuple[Event, Network],
) -> None:
    event, network = event_and_network
    short = Event(
        event.event_id,
        event.coa_time,
        event.series[:, 180:225],
        event.epochs + 180 / network.sampling_frequency,
        event.snr,
        event.bandwidth,
        event.horizon,
    )

    with pytest.raises(ValueError, match="BAYESTAR needs"):
        localize_event(short, network)


@pytest.mark.slow
def test_zero_noise_map_contains_the_truth(
    event_and_network: tuple[Event, Network], tmp_path: Path
) -> None:
    from ligo.skymap.io import fits

    event, network = event_and_network

    skymap = localize_event(event, network)

    assert len(skymap) == 19200
    assert set(skymap.colnames) == {
        "UNIQ", "PROBDENSITY", "DISTMU", "DISTSIGMA", "DISTNORM",
    }  # fmt: skip
    path = tmp_path / "map.multiorder.fits"
    fits.write_sky_map(str(path), skymap, moc=True)
    statistics = map_statistics(
        path, right_ascension=1.2, declination=-0.4, distance=1500.0
    )
    # Without noise the truth sits at the peak of the likelihood.
    assert statistics["searched_prob"] < 0.5
    assert statistics["searched_area"] < statistics["area_90"]
    assert statistics["distance_q05"] < 1500.0 < statistics["distance_q95"]
    assert 0.0 <= statistics["distance_lost_prob"] < 0.05
    assert statistics["log_bsn"] > 100


def test_zero_noise_posterior_prefers_the_true_inclination(
    event_and_network: tuple[Event, Network],
) -> None:
    """BAYESTAR's signal model fixes the handedness of the inclination.

    The population row is injected at inclination 0.5 rad, so the posterior
    evaluated at the true sky position, distance, polarization and time must
    beat the same point at ``pi - 0.5``.
    """
    event, network = event_and_network
    inputs = condition_event(event, network)
    right_ascension, declination = 1.2, -0.4
    delays = np.array(
        [
            lal.TimeDelayFromEarthCenter(
                location, right_ascension, declination, lal.LIGOTimeGPS(event.coa_time)
            )
            for location in network.locations
        ]
    )
    # The conditioned frame sits at the weighted-mean detector position, and
    # the posterior counts time from the start of the window, which is half the
    # series length before the weighted-mean arrival time.
    half_window = 0.5 * (inputs.snrs.shape[1] - 1) / network.sampling_frequency
    arrival = np.average(delays, weights=inputs.weights) - inputs.mean_toa + half_window

    def log_posterior(inclination: float) -> float:
        return float(
            log_posterior_toa_phoa_snr(
                right_ascension,
                np.sin(declination),
                1500.0,
                np.cos(inclination),
                2 * 0.3,
                arrival,
                network.prior.min_distance,
                network.prior.max_distance,
                network.prior.distance_power,
                network.prior.cosmology,
                inputs.gmst,
                network.sampling_frequency,
                inputs.first_samples,
                inputs.snrs,
                np.asarray(network.responses, dtype=np.float32),
                inputs.locations,
                np.asarray(event.horizon, dtype=float),
                network.rescale_loglikelihood,
            )
        )

    assert log_posterior(0.5) > log_posterior(np.pi - 0.5)
