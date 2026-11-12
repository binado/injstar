from pathlib import Path

import lal
import numpy as np
import pytest
from pydantic import ValidationError

from injstar.network import DetectorGeometry, DetectorRegistry, load_detector_config
from injstar.psd import NOISE_CURVES_BASE_DIR


@pytest.mark.parametrize("name", ["H1", "L1", "V1", "K1"])
def test_packaged_geometry_matches_lal(registry: DetectorRegistry, name: str) -> None:
    geometry = registry.detectors[name].geometry
    reference = lal.cached_detector_by_prefix[name]

    np.testing.assert_allclose(geometry.location, reference.location, atol=1e-3)
    # LAL tabulates arm directions independently of the geodetic angles.
    np.testing.assert_allclose(geometry.response_tensor, reference.response, atol=3e-3)


def test_response_tensor_of_sixty_degree_arms() -> None:
    geometry = DetectorGeometry(
        latitude_rad=0.0,
        longitude_rad=0.0,
        elevation_m=0.0,
        xarm_azimuth_rad=0.0,
        yarm_azimuth_rad=np.pi / 3,
    )
    # At (0, 0): north = z, east = y. x arm along z, y arm 60 degrees towards y.
    x = np.array([0.0, 0.0, 1.0])
    y = np.array([0.0, np.sin(np.pi / 3), np.cos(np.pi / 3)])
    expected = (np.outer(x, x) - np.outer(y, y)) / 2

    np.testing.assert_allclose(geometry.response_tensor, expected, atol=1e-15)
    assert np.trace(geometry.response_tensor) == pytest.approx(0.0, abs=1e-15)


def test_arrays_follow_declared_order(registry: DetectorRegistry) -> None:
    names, responses, locations = registry.arrays(["L1", "E1", "H1"])

    assert names == ("L1", "E1", "H1")
    assert responses.shape == (3, 3, 3)
    assert locations.shape == (3, 3)
    np.testing.assert_array_equal(
        locations[1], registry.detectors["E1"].geometry.location
    )


def test_unknown_network_and_detector_are_rejected(registry: DetectorRegistry) -> None:
    with pytest.raises(ValueError, match="unknown network"):
        registry.members("nowhere")
    with pytest.raises(ValueError, match="undefined detectors"):
        registry.members(["H1", "X9"])
    with pytest.raises(ValueError, match="distinct"):
        registry.members(["H1", "H1"])


def test_user_layer_adds_detector_and_network(tmp_path: Path) -> None:
    curve = np.loadtxt(NOISE_CURVES_BASE_DIR / "AplusDesign_psd.txt")
    np.savetxt(
        tmp_path / "mine_asd.txt", np.column_stack((curve[:, 0], np.sqrt(curve[:, 1])))
    )
    layer = tmp_path / "detectors.toml"
    layer.write_text(
        '[detectors.X1]\npsd_reference = "mine_asd.txt"\npsd_kind = "asd"\n'
        "[detectors.X1.geometry]\n"
        "latitude_rad = 0.5\nlongitude_rad = 1.0\nelevation_m = 10.0\n"
        "xarm_azimuth_rad = 0.0\nyarm_azimuth_rad = 1.0\n"
        '[detectors.H1]\npsd_reference = "kagra_128Mpc_psd.txt"\n'
        '[networks]\nmine = ["X1", "H1"]\n'
    )

    registry = load_detector_config([layer])

    assert registry.members("mine") == ("X1", "H1")
    assert registry.detectors["H1"].psd_reference == "kagra_128Mpc_psd.txt"
    # The packaged H1 geometry survives the override of its PSD.
    assert registry.detectors["H1"].geometry.elevation_m == pytest.approx(142.554)
    frequencies = np.array([1.0, 30.0, 100.0, 1e6])
    np.testing.assert_allclose(
        registry.detectors["X1"].sensitivity(frequencies),
        load_detector_config().detectors["H1"].sensitivity(frequencies),
        rtol=1e-12,
    )


def test_missing_psd_file_fails_validation(tmp_path: Path) -> None:
    layer = tmp_path / "detectors.toml"
    layer.write_text('[detectors.H1]\npsd_reference = "missing.txt"\n')

    with pytest.raises(ValidationError, match="Could not resolve PSD reference"):
        load_detector_config([layer])


def test_sensitivity_is_infinite_out_of_band(registry: DetectorRegistry) -> None:
    psd = registry.detectors["H1"].sensitivity(np.array([0.1, 100.0, 1e6]))

    assert np.isinf(psd[0]) and np.isinf(psd[2])
    assert 0 < psd[1] < 1e-40
