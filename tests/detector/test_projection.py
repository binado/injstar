import lal
import numpy as np
import pytest

from injstar.detector import projection


@pytest.mark.parametrize("name", ["H1", "L1", "V1"])
def test_antenna_pattern_and_delay_match_lal(name: str) -> None:
    detector = lal.cached_detector_by_prefix[name]
    rng = np.random.default_rng(1)
    ra = rng.uniform(0, 2 * np.pi, 20)
    dec = np.arcsin(rng.uniform(-1, 1, 20))
    psi = rng.uniform(0, np.pi, 20)
    time = lal.LIGOTimeGPS(1400000000)
    gmst = lal.GreenwichMeanSiderealTime(time)

    plus, cross = projection.antenna_pattern(
        detector.response.astype(float),
        gmst,
        right_ascension=ra,
        declination=dec,
        polarization_angle=psi,
    )
    delay = projection.time_delay_from_geocenter(
        detector.location, gmst, right_ascension=ra, declination=dec
    )

    expected = np.array(
        [
            lal.ComputeDetAMResponse(detector.response, a, d, p, gmst)
            for a, d, p in zip(ra, dec, psi, strict=True)
        ]
    )
    expected_delay = [
        lal.TimeDelayFromEarthCenter(detector.location, a, d, time)
        for a, d in zip(ra, dec, strict=True)
    ]
    np.testing.assert_allclose(plus, expected[:, 0], atol=1e-7)
    np.testing.assert_allclose(cross, expected[:, 1], atol=1e-7)
    np.testing.assert_allclose(delay, expected_delay, atol=1e-9)
