from pathlib import Path

import lal
import numpy as np
import pytest

from injstar.localize import Event, Network, localize_event, pack
from injstar.products import map_statistics


def test_pack_follows_bayestar_conditioning(
    event_and_network: tuple[Event, Network],
) -> None:
    event, network = event_and_network

    packed = pack(event, network)

    weights = np.square(event.snr * event.bandwidth)
    length = packed.snrs.shape[1]
    assert length % 2 == 1 and length <= event.series.shape[1]
    # Positions and arrival times are centred on their weighted means.
    np.testing.assert_allclose(
        np.average(packed.locations, weights=weights, axis=0), 0.0, atol=1e-15
    )
    toas = packed.epochs + 0.5 * (length - 1) / network.sample_rate
    assert np.average(toas, weights=weights) == pytest.approx(0.0, abs=1e-12)
    # Centring moves nothing in absolute terms.
    np.testing.assert_allclose(
        packed.locations[0] - packed.locations[1],
        (network.locations[0] - network.locations[1]) / 299792458.0,
    )
    centre = event.series.shape[1] // 2
    absolute = float(packed.epoch) + toas
    expected = event.coa_time + event.epochs + centre / network.sample_rate
    np.testing.assert_allclose(absolute, expected, atol=2e-7)
    # The window covers the network's light-travel time plus BAYESTAR's margin.
    radius = np.linalg.norm(packed.locations, axis=1).max()
    assert length >= 2 * np.ceil((radius + 0.005) * network.sample_rate) - 1
    np.testing.assert_allclose(packed.snrs[:, length // 2, 0], event.snr, rtol=1e-12)
    assert packed.gmst == pytest.approx(lal.GreenwichMeanSiderealTime(packed.epoch))


def test_pack_rejects_a_short_series(event_and_network: tuple[Event, Network]) -> None:
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
        pack(short, network)


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
    assert statistics["log_bsn"] > 100
