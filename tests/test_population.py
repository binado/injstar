from pathlib import Path

import h5py
import numpy as np
import pytest

from injstar.population import CANONICAL_COLUMNS, read_population, simulation_events


def test_csv_gives_canonical_columns(population_csv: Path) -> None:
    population = read_population(population_csv)

    assert tuple(population) == CANONICAL_COLUMNS
    np.testing.assert_array_equal(population["event_id"], [0, 1, 2])
    assert population["event_id"].dtype == np.uint64
    assert population["distance"][0] == 1500.0


def test_hdf5_structured_dataset_matches_csv(
    population_csv: Path, tmp_path: Path
) -> None:
    expected = read_population(population_csv)
    names = CANONICAL_COLUMNS[1:]
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


def test_snr_columns_are_carried_not_used(population_csv: Path, tmp_path: Path) -> None:
    lines = population_csv.read_text().splitlines()
    path = tmp_path / "with_snr.csv"
    path.write_text(
        "\n".join([lines[0] + ",snr", *(line + ",999.0" for line in lines[1:])]) + "\n"
    )

    population = read_population(path)

    np.testing.assert_array_equal(population["population_snr"], [999.0] * 3)
    assert "snr" not in population


def test_luminosity_distance_alias(population_csv: Path, tmp_path: Path) -> None:
    path = tmp_path / "alias.csv"
    path.write_text(
        population_csv.read_text().replace(",distance,", ",luminosity_distance,")
    )

    assert read_population(path)["distance"][0] == 1500.0


def test_source_frame_masses_are_redshifted(tmp_path: Path) -> None:
    path = tmp_path / "source.csv"
    path.write_text(
        "source_frame_mass_1,source_frame_mass_2,redshift,right_ascension,"
        "declination,inclination,polarization_angle,coa_time\n"
        "30.0,20.0,0.5,1.0,0.2,0.3,0.4,1400000000.0\n"
    )

    population = read_population(path, cosmology="Planck18")

    assert population["detector_frame_mass_1"][0] == pytest.approx(45.0)
    assert population["detector_frame_mass_2"][0] == pytest.approx(30.0)
    assert population["distance"][0] == pytest.approx(2918, rel=0.01)


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


def test_simulation_events_mass_parameters(population_csv: Path) -> None:
    population = read_population(population_csv)

    events = simulation_events(population, np.zeros(3))

    assert events["chirp_mass"][0] == pytest.approx(28.0956, rel=1e-4)
    assert events["symmetric_mass_ratio"][0] == pytest.approx(36 * 29 / 65**2)
