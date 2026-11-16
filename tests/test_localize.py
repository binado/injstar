from dataclasses import replace
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
    short = replace(
        event,
        series=event.series[:, 180:225],
        epochs=event.epochs + 180 / network.sampling_frequency,
    )

    with pytest.raises(ValueError, match="BAYESTAR needs"):
        localize_event(short, network, max_distance=5000.0)


@pytest.mark.slow
def test_zero_noise_map_contains_the_truth(
    event_and_network: tuple[Event, Network],
) -> None:
    event, network = event_and_network

    skymap = localize_event(event, network, max_distance=5000.0)

    assert len(skymap) == 19200
    assert set(skymap.colnames) == {
        "UNIQ", "PROBDENSITY", "DISTMU", "DISTSIGMA", "DISTNORM",
    }  # fmt: skip
    statistics = map_statistics(
        skymap,
        row_index=event.row_index,
        right_ascension=1.2,
        declination=-0.4,
        distance=1500.0,
    )
    # Without noise the truth sits at the peak of the likelihood.
    assert statistics.searched_prob < 0.5
    assert statistics.searched_area < statistics.area_90
    assert statistics.distance_q05 < 1500.0 < statistics.distance_q95
    assert 0.0 <= statistics.distance_lost_prob < 0.05
    assert statistics.log_bsn > 100


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
                0.0,
                5000.0,
                2,
                False,
                inputs.gmst,
                network.sampling_frequency,
                inputs.first_samples,
                inputs.snrs,
                np.asarray(network.responses, dtype=np.float32),
                inputs.locations,
                np.asarray(event.horizon, dtype=float),
                1.0,
            )
        )

    assert log_posterior(0.5) > log_posterior(np.pi - 0.5)


@pytest.mark.parametrize("failure", ["statistics", "serialization"])
def test_failed_summary_does_not_publish_map(
    event_and_network: tuple[Event, Network],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    from unittest.mock import Mock

    from astropy.table import Table

    from injstar.io import skymap_path
    from injstar.localize import localize_to_file
    from injstar.products import LocalizationSummary

    event, network = event_and_network
    skymap = Table(
        {"UNIQ": np.arange(4, 16), "PROBDENSITY": np.full(12, 1 / (4 * np.pi))}
    )
    monkeypatch.setattr("injstar.localize.localize_event", Mock(return_value=skymap))
    summary = Mock(spec=LocalizationSummary)
    summary.dump_fits.side_effect = ValueError("failed summary")
    statistics = Mock(return_value=summary)
    if failure == "statistics":
        statistics.side_effect = ValueError("failed summary")
    monkeypatch.setattr("injstar.localize.map_statistics", statistics)

    with pytest.raises(ValueError, match="failed summary"):
        localize_to_file(event, network, tmp_path, max_distance=5000.0)

    assert not skymap_path(tmp_path, event.row_index).exists()
    assert not list(tmp_path.rglob("*.partial.fits"))
