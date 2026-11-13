import numpy as np
import pytest
from ligo.skymap import distance

from injstar.products import DISTANCE_QUANTILES, _distance_statistics, statistic_names


@pytest.fixture
def layer() -> dict[str, np.ndarray]:
    """Five pixels with a valid distance layer around 500 to 700 Mpc."""
    mean = np.array([500.0, 550.0, 600.0, 650.0, 700.0])
    std = np.array([40.0, 50.0, 60.0, 50.0, 40.0])
    mu, sigma, norm = distance.moments_to_parameters(mean, std)
    return {
        "prob": np.array([0.1, 0.3, 0.3, 0.2, 0.1]),
        "mu": mu,
        "sigma": sigma,
        "norm": norm,
    }


def _statistics(
    layer: dict[str, np.ndarray],
) -> tuple[float, np.ndarray]:
    return _distance_statistics(
        layer["prob"],
        layer["mu"],
        layer["sigma"],
        layer["norm"],
        distance.marginal_ppf,
    )


def _with_failed(
    layer: dict[str, np.ndarray], prob: float = 0.4
) -> dict[str, np.ndarray]:
    """Append a pixel as BAYESTAR writes one it could not evaluate."""
    return {
        "prob": np.append(layer["prob"] * (1 - prob), prob),
        "mu": np.append(layer["mu"], np.inf),
        "sigma": np.append(layer["sigma"], 1.0),
        "norm": np.append(layer["norm"], 0.0),
    }


def test_distance_statistics_without_failures_lose_nothing(
    layer: dict[str, np.ndarray],
) -> None:
    lost, quantiles = _statistics(layer)

    assert lost == 0.0
    assert np.all(np.diff(quantiles) > 0)
    assert 400.0 < quantiles[0] < quantiles[-1] < 800.0


def test_distance_statistics_report_the_probability_of_failed_pixels(
    layer: dict[str, np.ndarray],
) -> None:
    lost, _ = _statistics(_with_failed(layer, prob=0.4))

    assert lost == pytest.approx(0.4)


def test_distance_statistics_with_failures_equal_those_of_the_valid_pixels(
    layer: dict[str, np.ndarray],
) -> None:
    _, expected = _statistics(layer)

    _, quantiles = _statistics(_with_failed(layer))

    np.testing.assert_allclose(quantiles, expected)


def test_distance_statistics_ignore_the_values_in_failed_pixels(
    layer: dict[str, np.ndarray],
) -> None:
    clean = _with_failed(layer)
    garbled = clean | {"mu": np.append(clean["mu"][:-1], np.nan)}

    np.testing.assert_allclose(_statistics(garbled)[1], _statistics(clean)[1])


def test_distance_statistics_with_every_pixel_failed_are_nan() -> None:
    failed = {
        "prob": np.array([0.5, 0.5]),
        "mu": np.full(2, np.inf),
        "sigma": np.ones(2),
        "norm": np.zeros(2),
    }

    lost, quantiles = _statistics(failed)

    assert lost == 1.0
    assert quantiles.shape == (len(DISTANCE_QUANTILES),)
    assert np.all(np.isnan(quantiles))


def test_summary_columns_include_the_lost_probability() -> None:
    names = statistic_names()

    assert "distance_lost_prob" in names
    assert names.index("distance_lost_prob") > names.index("distance_q95")


def test_distance_statistics_lost_probability_is_never_negative(
    layer: dict[str, np.ndarray],
) -> None:
    # Summing probabilities that total one can round to a hair above it.
    crowded = layer | {"prob": layer["prob"] * (1 + 1e-15)}

    lost, _ = _statistics(crowded)

    assert lost >= 0.0
