from pathlib import Path

import pytest

from injstar.localize import Event, Network, localize_event
from injstar.products import map_statistics


def test_localize_event_rejects_a_short_series(
    event_and_network: tuple[Event, Network],
) -> None:
    event, network = event_and_network
    short = Event(
        event.event_id,
        event.coa_time,
        event.series[:, 180:225],
        event.epochs + 180 / network.sample_rate,
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
