from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from injstar import products
from injstar.cli import app
from injstar.config import RunConfig, load_run_config

runner = CliRunner()


def invoke(*args: object) -> tuple[int, str]:
    result = runner.invoke(app, [str(arg) for arg in args])
    return result.exit_code, result.output


def use_named_network(config_file: Path) -> None:
    detectors = config_file.parent / "network.toml"
    detectors.write_text('[networks]\nregression = ["H1", "L1", "V1"]\n')
    config_file.write_text(
        config_file.read_text().replace(
            'network = ["H1", "L1", "V1"]',
            'network = "regression"\ndetector_config = ["network.toml"]',
        )
    )


def test_named_network_matches_explicit_detectors(config_file: Path) -> None:
    explicit = config_file.parent / "explicit"
    named = config_file.parent / "named"
    code, output = invoke("snr", config_file, "--output-dir", explicit)
    assert code == 0, output

    use_named_network(config_file)
    code, output = invoke("snr", config_file, "--output-dir", named)
    assert code == 0, output

    expected, _ = products.read_table(explicit / "injections.h5")
    actual, _ = products.read_table(named / "injections.h5")
    assert actual.keys() == expected.keys()
    for name, values in expected.items():
        np.testing.assert_array_equal(actual[name], values)


def test_snr_writes_injections_and_no_maps(config_file: Path) -> None:
    code, output = invoke("snr", config_file)

    assert code == 0, output
    assert "2 of 3 injections detected" in output
    out = config_file.parent / "out"
    assert sorted(path.name for path in out.iterdir()) == [
        "config.json", "injections.h5",
    ]  # fmt: skip
    columns, _ = products.read_table(out / "injections.h5")
    for name in ("expected_snr_H1", "observed_snr_V1", "horizon_L1", "network_snr"):
        assert columns[name].shape == (3,)
    np.testing.assert_array_equal(columns["detected"], [True, False, True])


def test_snr_records_the_config_it_ran(config_file: Path) -> None:
    invoke("snr", config_file)

    recorded = RunConfig.model_validate_json(
        (config_file.parent / "out" / "config.json").read_text()
    )
    assert recorded == load_run_config(config_file)


def test_snr_ignores_snr_columns_in_the_population(
    config_file: Path, population_csv: Path, tmp_path: Path
) -> None:
    invoke("snr", config_file, "--output-dir", tmp_path / "plain")
    lines = population_csv.read_text().splitlines()
    population_csv.write_text(
        "\n".join([lines[0] + ",snr", *(line + ",999.0" for line in lines[1:])]) + "\n"
    )
    code, output = invoke("snr", config_file, "--output-dir", tmp_path / "bogus")

    assert code == 0, output
    plain, _ = products.read_table(tmp_path / "plain" / "injections.h5")
    bogus, _ = products.read_table(tmp_path / "bogus" / "injections.h5")
    np.testing.assert_array_equal(bogus["population_snr"], [999.0] * 3)
    for name, values in plain.items():
        np.testing.assert_array_equal(bogus[name], values)


def test_snr_reuses_a_matching_directory_and_refuses_another(
    config_file: Path,
) -> None:
    invoke("snr", config_file)
    target = config_file.parent / "out" / "injections.h5"
    written = target.stat().st_mtime_ns

    assert invoke("snr", config_file)[0] == 0
    assert target.stat().st_mtime_ns == written

    code, output = invoke("snr", config_file, "--seed", 5)
    assert code == 1
    assert "different config" in output

    assert invoke("snr", config_file, "--seed", 5, "--overwrite")[0] == 0
    assert target.stat().st_mtime_ns != written


def test_foreign_directory_is_never_deleted(config_file: Path) -> None:
    out = config_file.parent / "out"
    out.mkdir()
    (out / "thesis.tex").write_text("precious")

    code, output = invoke("snr", config_file, "--overwrite")

    assert code == 1
    assert "refusing to delete" in output
    assert (out / "thesis.tex").read_text() == "precious"


def test_invalid_config_reports_an_error(config_file: Path) -> None:
    config_file.write_text(config_file.read_text().replace("5000.0", "-1.0"))

    code, output = invoke("snr", config_file)

    assert code == 1
    assert "max_distance" in output


@pytest.mark.slow
def test_run_resume_and_summarise(config_file: Path) -> None:
    out = config_file.parent / "out"
    assert invoke("snr", config_file)[0] == 0

    code, output = invoke("run", config_file, "--workers", 2)

    assert code == 0, output
    maps = sorted(path.name for path in (out / "skymaps").iterdir())
    assert maps == ["0.multiorder.fits", "2.multiorder.fits"]
    first, _ = products.read_table(out / "summary.h5")
    np.testing.assert_array_equal(first["localized"], [True, False, True])
    assert np.isnan(first["area_90"][1])
    assert first["skymap"][0] == "skymaps/0.multiorder.fits"
    assert np.all(first["area_90"][[0, 2]] > first["area_50"][[0, 2]])

    # A finished directory is not silently redone.
    assert invoke("run", config_file)[0] == 1
    assert invoke("run", config_file, "--resume", "--overwrite")[0] == 1

    kept = (out / "skymaps" / "0.multiorder.fits").stat().st_mtime_ns
    (out / "skymaps" / "2.multiorder.fits").unlink()
    (out / "summary.h5").unlink()
    code, output = invoke("run", config_file, "--resume")

    assert code == 0, output
    assert (out / "skymaps" / "0.multiorder.fits").stat().st_mtime_ns == kept
    resumed, _ = products.read_table(out / "summary.h5")
    for name, values in first.items():
        np.testing.assert_array_equal(resumed[name], values)

    (out / "summary.h5").unlink()
    assert invoke("summarise", out)[0] == 0
    rebuilt, _ = products.read_table(out / "summary.h5")
    for name, values in first.items():
        np.testing.assert_array_equal(rebuilt[name], values)
