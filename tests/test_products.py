from pathlib import Path

import numpy as np
import pytest
from ligo.skymap import distance

from injstar.products import (
    DISTANCE_QUANTILES,
    LocalizationSummary,
    _distance_statistics,
    statistic_names,
)

from .tolerances import assert_close


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

    assert_close(quantiles, expected)


def test_distance_statistics_ignore_the_values_in_failed_pixels(
    layer: dict[str, np.ndarray],
) -> None:
    clean = _with_failed(layer)
    garbled = clean | {"mu": np.append(clean["mu"][:-1], np.nan)}

    assert_close(_statistics(garbled)[1], _statistics(clean)[1])


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

    assert names == (
        "area_50",
        "area_90",
        "searched_area",
        "searched_prob",
        "searched_prob_dist",
        "searched_prob_vol",
        "offset",
        "distance_q05",
        "distance_q50",
        "distance_q95",
        "distance_lost_prob",
        "distmean",
        "diststd",
        "log_bci",
        "log_bsn",
    )


def test_distance_statistics_lost_probability_is_never_negative(
    layer: dict[str, np.ndarray],
) -> None:
    # Summing probabilities that total one can round to a hair above it.
    crowded = layer | {"prob": layer["prob"] * (1 + 1e-15)}

    lost, _ = _statistics(crowded)

    assert lost >= 0.0


@pytest.fixture
def summary() -> LocalizationSummary:
    return LocalizationSummary(
        row_index=10,
        area_50=1.0,
        area_90=2.0,
        searched_area=0.5,
        searched_prob=0.2,
        searched_prob_dist=0.3,
        searched_prob_vol=0.4,
        offset=0.1,
        distance_q05=400.0,
        distance_q50=600.0,
        distance_q95=800.0,
        distance_lost_prob=0.0,
        distmean=600.0,
        diststd=50.0,
        log_bci=np.nan,
        log_bsn=100.0,
    )


@pytest.fixture
def map_file(tmp_path: Path) -> Path:
    from astropy.table import Table
    from ligo.skymap.io import write_sky_map

    path = tmp_path / "map.fits"
    skymap = Table(
        {"UNIQ": np.arange(4, 16), "PROBDENSITY": np.full(12, 1 / (4 * np.pi))}
    )
    write_sky_map(str(path), skymap, moc=True)
    return path


def test_summary_fits_roundtrip_and_replacement(
    map_file: Path, summary: LocalizationSummary
) -> None:
    from dataclasses import asdict, replace

    from astropy.io import fits
    from ligo.skymap.io import read_sky_map

    before = read_sky_map(str(map_file), moc=True, hdu=1)
    for value in (summary, replace(summary, row_index=2, area_90=3.0)):
        value.dump_fits(map_file)
        actual = LocalizationSummary.from_fits(map_file)
        np.testing.assert_equal(
            list(asdict(actual).values()), list(asdict(value).values())
        )
        with fits.open(map_file) as hdus:
            assert len(hdus) == 3
            assert hdus[2].name == "SUMMARY"
            assert hdus[2].columns.formats == ["K", *(["D"] * len(statistic_names()))]
        after = read_sky_map(str(map_file), moc=True, hdu=1)
        assert after.meta == before.meta
        for name in before.colnames:
            np.testing.assert_array_equal(after[name], before[name])


def test_missing_summary_fails_clearly(map_file: Path) -> None:
    with pytest.raises(ValueError, match="has no SUMMARY extension"):
        LocalizationSummary.from_fits(map_file)


def test_collect_stored_summaries(
    tmp_path: Path, map_file: Path, summary: LocalizationSummary
) -> None:
    import csv
    import shutil
    from dataclasses import asdict, replace

    from injstar.io import skymap_path
    from injstar.products import build_summary

    values = [summary, replace(summary, row_index=2, area_90=4.0)]
    for value in values:
        path = skymap_path(tmp_path, value.row_index)
        path.parent.mkdir(exist_ok=True)
        shutil.copyfile(map_file, path)
        value.dump_fits(path)
    # Incomplete worker files never contribute a row.
    (path.parent / ".3.multiorder.fits.123.partial.fits").write_text("incomplete")
    target = build_summary(tmp_path)
    assert target == tmp_path / "summary.csv"
    with target.open(newline="") as stream:
        reader = csv.DictReader(stream)
        assert reader.fieldnames == ["row_index", *statistic_names()]
        rows = list(reader)
    assert [int(row["row_index"]) for row in rows] == [2, 10]
    for row, value in zip(rows, reversed(values), strict=True):
        np.testing.assert_equal(
            [float(item) for item in row.values()], list(asdict(value).values())
        )
        assert row["log_bci"] == "nan"


def test_collect_rejects_a_map_without_summary(tmp_path: Path, map_file: Path) -> None:
    import shutil

    from injstar.io import skymap_path
    from injstar.products import build_summary

    path = skymap_path(tmp_path, 0)
    path.parent.mkdir()
    shutil.copyfile(map_file, path)
    with pytest.raises(ValueError, match="has no SUMMARY extension"):
        build_summary(tmp_path)


@pytest.fixture
def summary_directories(tmp_path: Path, summary: LocalizationSummary) -> list[Path]:
    import csv
    from dataclasses import asdict, replace

    directories = [tmp_path / name for name in ("a", "b", "empty")]
    for directory, indices in zip(directories, [[10, 2], [100, 3], []], strict=True):
        directory.mkdir()
        with (directory / "summary.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(
                stream, fieldnames=("row_index", *statistic_names())
            )
            writer.writeheader()
            writer.writerows(
                asdict(replace(summary, row_index=index)) for index in indices
            )
    return directories


def test_collect_chunk_summaries(
    summary_directories: list[Path], tmp_path: Path
) -> None:
    import csv

    from injstar.products import collect_summaries

    target = tmp_path / "combined.csv"
    assert collect_summaries(reversed(summary_directories), target) == target
    with target.open() as stream:
        assert [int(row["row_index"]) for row in csv.DictReader(stream)] == [
            2,
            3,
            10,
            100,
        ]
    before = target.read_bytes()
    with pytest.raises(ValueError, match="exists"):
        collect_summaries(summary_directories, target)
    assert target.read_bytes() == before
    collect_summaries([summary_directories[-1]], target, overwrite=True)
    with target.open() as stream:
        assert list(csv.DictReader(stream)) == []


@pytest.mark.parametrize(
    "problem", ["duplicates", "missing", "columns", "empty", "values"]
)
def test_collect_validates_before_writing(
    summary_directories: list[Path], tmp_path: Path, problem: str
) -> None:
    from injstar.products import collect_summaries

    directories = summary_directories
    if problem == "duplicates":
        directories = [*directories, directories[0]]
    elif problem == "missing":
        (directories[1] / "summary.csv").unlink()
    elif problem == "columns":
        (directories[1] / "summary.csv").write_text("row_index,bogus\n")
    elif problem == "values":
        path = directories[1] / "summary.csv"
        path.write_text(path.read_text().replace("100,1.0", "100,invalid"))
    else:
        directories = []
    output = tmp_path / "combined.csv"
    output.write_text("keep")
    with pytest.raises((ValueError, FileNotFoundError)):
        collect_summaries(directories, output, overwrite=True)
    assert output.read_text() == "keep"
