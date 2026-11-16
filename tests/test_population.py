from pathlib import Path

import h5py
import numpy as np
import pytest

from injstar.population import CANONICAL_COLUMNS, read_population, simulation_events


def test_csv_gives_canonical_columns(population_csv: Path) -> None:
    population = read_population(population_csv)

    assert tuple(population) == CANONICAL_COLUMNS
    assert population["distance"][0] == 1500.0


def test_hdf5_structured_dataset_matches_csv(
    population_csv: Path, tmp_path: Path
) -> None:
    expected = read_population(population_csv)
    names = CANONICAL_COLUMNS
    table = np.zeros(3, dtype=[(name, "f8") for name in names])
    for name in names:
        table[name] = expected[name]
    path = tmp_path / "population.h5"
    with h5py.File(path, "w") as file:
        file.create_dataset("data", data=table)
        file.create_group("metadata")

    population = read_population(path)

    for name in CANONICAL_COLUMNS:
        np.testing.assert_array_equal(population[name], expected[name])


def test_extra_columns_keep_names_and_row_order(
    population_csv: Path, tmp_path: Path
) -> None:
    lines = population_csv.read_text().splitlines()
    path = tmp_path / "extra.csv"
    path.write_text(
        "\n".join(
            [
                lines[0] + ",label",
                *(
                    line + "," + label
                    for line, label in zip(lines[1:], ["b", "a", "c"], strict=True)
                ),
            ]
        )
        + "\n"
    )
    population = read_population(path)
    np.testing.assert_array_equal(population["label"], ["b", "a", "c"])
    np.testing.assert_array_equal(population["detector_frame_mass_1"], [36, 30, 1.4])


def test_hdf5_group_matches_csv(population_csv: Path, tmp_path: Path) -> None:
    expected = read_population(population_csv)
    path = tmp_path / "population.hdf5"
    with h5py.File(path, "w") as file:
        group = file.create_group("data")
        for name, values in expected.items():
            group.create_dataset(name, data=values)
    actual = read_population(path)
    for name in expected:
        np.testing.assert_array_equal(actual[name], expected[name])


@pytest.mark.parametrize(
    ("old", "new"),
    [
        (",distance,", ",luminosity_distance,"),
        ("detector_frame_mass_1", "source_frame_mass_1"),
        ("detector_frame_mass_2", "source_frame_mass_2"),
    ],
)
def test_canonical_columns_required(
    population_csv: Path, tmp_path: Path, old: str, new: str
) -> None:
    path = tmp_path / "alias.csv"
    path.write_text(population_csv.read_text().replace(old, new))
    with pytest.raises(ValueError, match="lacks the columns"):
        read_population(path)


def test_event_id_is_preserved(population_csv: Path) -> None:
    lines = population_csv.read_text().splitlines()
    population_csv.write_text(
        "\n".join([lines[0] + ",event_id", *(line + ",1" for line in lines[1:])]) + "\n"
    )
    np.testing.assert_array_equal(
        read_population(population_csv)["event_id"], [1, 1, 1]
    )


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        (",coa_time", ",time", "lacks the columns"),
        ("36.0,29.0", "-36.0,29.0", "must be positive"),
        ("1500.0", "nan", "non-finite"),
    ],
)
def test_bad_populations_are_rejected(
    population_csv: Path, tmp_path: Path, old: str, new: str, message: str
) -> None:
    path = tmp_path / "bad.csv"
    path.write_text(population_csv.read_text().replace(old, new, 1))

    with pytest.raises(ValueError, match=message):
        read_population(path)


@pytest.mark.parametrize(
    "name",
    ["spin_1x", "spin_1y", "spin_2x", "spin_2y", "lambda_1", "lambda_2", "coa_phase"],
)
def test_optional_columns_default_to_zero(population_csv: Path, name: str) -> None:
    population = read_population(population_csv)

    np.testing.assert_array_equal(population[name], np.zeros(3))


def test_simulation_events_pass_columns_through(population_csv: Path) -> None:
    population = read_population(population_csv)
    gmst = np.array([0.1, 0.2, 0.3])

    events = simulation_events(population, gmst)

    assert "coa_time" not in events
    np.testing.assert_array_equal(events["gmst"], gmst)
    for name in ("detector_frame_mass_1", "distance", "inclination"):
        np.testing.assert_array_equal(events[name], population[name])


@pytest.fixture(params=["csv", "structured", "grouped"])
def ranged_population(
    request: pytest.FixtureRequest, population_csv: Path, tmp_path: Path
) -> Path:
    """Three rows with optional columns absent and an extra text column."""
    lines = population_csv.read_text().splitlines()
    population_csv.write_text(
        "\n".join(
            [
                lines[0] + ",label",
                *(
                    line + "," + label
                    for line, label in zip(lines[1:], ["b", "a", "c"], strict=True)
                ),
            ]
        )
        + "\n"
    )
    if request.param == "csv":
        return population_csv
    data = np.genfromtxt(
        population_csv, delimiter=",", names=True, dtype=None, encoding="utf-8"
    )
    data = data.astype(
        [
            (name, "S1" if name == "label" else data.dtype[name])
            for name in data.dtype.names or ()
        ]
    )
    path = tmp_path / "population.h5"
    with h5py.File(path, "w") as file:
        if request.param == "structured":
            file.create_dataset("data", data=data)
        else:
            group = file.create_group("data")
            for name in data.dtype.names or ():
                group.create_dataset(name, data=data[name])
    return path


@pytest.mark.parametrize(("start", "stop"), [(0, None), (1, 2), (1, None), (2, 100)])
def test_population_range(
    ranged_population: Path, start: int, stop: int | None
) -> None:
    full = read_population(ranged_population)
    selected = read_population(ranged_population, start=start, stop=stop)
    assert tuple(selected) == (*CANONICAL_COLUMNS, "label")
    for name in full:
        np.testing.assert_array_equal(selected[name], full[name][start:stop])
    np.testing.assert_array_equal(
        selected["lambda_1"], np.zeros(len(selected["label"]))
    )


@pytest.mark.parametrize(
    ("start", "stop"), [(-1, None), (0, -1), (1, 1), (2, 1), (3, None), (10, 20)]
)
def test_invalid_population_range(
    ranged_population: Path, start: int, stop: int | None
) -> None:
    with pytest.raises(ValueError, match="range|no rows"):
        read_population(ranged_population, start=start, stop=stop)


def test_csv_range_infers_selected_values_only(population_csv: Path) -> None:
    population_csv.write_text(population_csv.read_text().replace("1500.0", "invalid"))
    selected = read_population(population_csv, start=1, stop=2)
    np.testing.assert_allclose(selected["distance"], [40000.0])


def test_population_validates_only_selected_range(ranged_population: Path) -> None:
    if ranged_population.suffix == ".csv":
        ranged_population.write_text(
            ranged_population.read_text().replace("1500.0", "nan")
        )
    else:
        with h5py.File(ranged_population, "r+") as file:
            source = file["data"]
            if isinstance(source, h5py.Dataset):
                row = source[0]
                row["distance"] = np.nan
                source[0] = row
            else:
                source["distance"][0] = np.nan
    np.testing.assert_allclose(
        read_population(ranged_population, start=1, stop=2)["distance"], [40000.0]
    )
    with pytest.raises(ValueError, match="non-finite"):
        read_population(ranged_population)
