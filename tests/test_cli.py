import csv
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from gwmock_pop.loaders.file_loader import read_population_catalogue
from typer.testing import CliRunner

from injstar.cli import app
from injstar.config import InjectConfig, load_inject_config
from injstar.config.inject import Inputs
from injstar.dataset import InjectionResult
from injstar.localize import Event, Network
from injstar.products import LocalizationSummary, build_summary, statistic_names

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
    code, output = invoke("inject", config_file, "--output-dir", explicit)
    assert code == 0, output

    use_named_network(config_file)
    code, output = invoke("inject", config_file, "--output-dir", named)
    assert code == 0, output

    expected = read_population_catalogue(explicit / "injections.h5")
    actual = read_population_catalogue(named / "injections.h5")
    assert actual.keys() == expected.keys()
    for name, values in expected.items():
        np.testing.assert_array_equal(actual[name], values)


def test_inject_writes_injections_and_no_maps(config_file: Path) -> None:
    code, output = invoke("inject", config_file)

    assert code == 0, output
    assert "2 of 3 injections detected" in output
    out = config_file.parent / "out"
    assert sorted(path.name for path in out.iterdir()) == [
        "config.json", "injection_results.h5", "injections.h5",
    ]  # fmt: skip
    columns = read_population_catalogue(out / "injections.h5")
    for name in ("expected_snr_H1", "observed_snr_V1", "horizon_L1", "network_snr"):
        assert columns[name].shape == (3,)
    np.testing.assert_array_equal(columns["detected"], [True, False, True])
    dataset = InjectionResult.read(out / "injection_results.h5")
    np.testing.assert_allclose(dataset.network_snr, columns["network_snr"])
    np.testing.assert_array_equal(dataset.detected, columns["detected"])
    assert "event_id" not in columns


def test_inject_records_the_config_it_ran(config_file: Path) -> None:
    invoke("inject", config_file)

    recorded = InjectConfig.model_validate_json(
        (config_file.parent / "out" / "config.json").read_text()
    )
    assert recorded == load_inject_config(config_file)


def test_inject_ignores_snr_columns_in_the_population(
    config_file: Path, population_csv: Path, tmp_path: Path
) -> None:
    invoke("inject", config_file, "--output-dir", tmp_path / "plain")
    lines = population_csv.read_text().splitlines()
    population_csv.write_text(
        "\n".join(
            [lines[0] + ",snr,event_id", *(line + ",999.0,42" for line in lines[1:])]
        )
        + "\n"
    )
    code, output = invoke("inject", config_file, "--output-dir", tmp_path / "bogus")

    assert code == 0, output
    plain = read_population_catalogue(tmp_path / "plain" / "injections.h5")
    bogus = read_population_catalogue(tmp_path / "bogus" / "injections.h5")
    np.testing.assert_array_equal(bogus["snr"], [999.0] * 3)
    np.testing.assert_array_equal(bogus["event_id"], [42] * 3)
    for name, values in plain.items():
        np.testing.assert_array_equal(bogus[name], values)


def test_inject_reuses_a_matching_directory_and_refuses_another(
    config_file: Path,
) -> None:
    invoke("inject", config_file)
    target = config_file.parent / "out" / "injections.h5"
    written = target.stat().st_mtime_ns

    assert invoke("inject", config_file)[0] == 0
    assert target.stat().st_mtime_ns == written

    code, output = invoke("inject", config_file, "--seed", 5)
    assert code == 1
    assert "different config" in output

    assert invoke("inject", config_file, "--seed", 5, "--overwrite")[0] == 0
    assert target.stat().st_mtime_ns != written


def test_foreign_directory_is_never_deleted(config_file: Path) -> None:
    out = config_file.parent / "out"
    out.mkdir()
    (out / "thesis.tex").write_text("precious")

    code, output = invoke("inject", config_file, "--overwrite")

    assert code == 1
    assert "refusing to delete" in output
    assert (out / "thesis.tex").read_text() == "precious"


def test_invalid_config_reports_an_error(config_file: Path) -> None:
    config_file.write_text(config_file.read_text() + "\nminimum_frequency = -1.0\n")

    code, output = invoke("inject", config_file)

    assert code == 1
    assert "minimum_frequency" in output


@pytest.mark.slow
def test_localize_resume_and_summary(
    config_file: Path, population_csv: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
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
    out = config_file.parent / "out"
    assert invoke("inject", config_file)[0] == 0

    def disabled(*args: object, **kwargs: object) -> None:
        pytest.fail("localization must reuse stored injection results")

    monkeypatch.setattr("injstar.config.inject._simulation_batches", disabled)
    monkeypatch.setattr("injstar.inject.inject_events", disabled)
    monkeypatch.setattr("injstar.waveform.waveform", disabled)
    localize_file = make_localize_file(config_file)
    code, output = invoke("localize", localize_file, "--workers", 2)

    assert code == 0, output
    out = config_file.parent / "localized"
    maps = sorted(path.name for path in (out / "skymaps").iterdir())
    assert maps == ["0.multiorder.fits", "2.multiorder.fits"]
    first = read_summary(out)
    assert tuple(first[0]) == ("row_index", *statistic_names())
    assert [int(row["row_index"]) for row in first] == [0, 2]
    for row in first:
        summary = LocalizationSummary.from_fits(
            out / "skymaps" / f"{row['row_index']}.multiorder.fits"
        )
        assert summary.area_90 > summary.area_50
        assert float(row["area_90"]) == summary.area_90

    assert invoke("localize", localize_file)[0] == 1
    assert invoke("localize", localize_file, "--resume", "--overwrite")[0] == 1

    kept = (out / "skymaps" / "0.multiorder.fits").stat().st_mtime_ns
    (out / "skymaps" / "2.multiorder.fits").unlink()
    (out / "summary.csv").unlink()
    code, output = invoke("localize", localize_file, "--resume")

    assert code == 0, output
    assert (out / "skymaps" / "0.multiorder.fits").stat().st_mtime_ns == kept
    assert read_summary(out) == first

    kept_maps = {path: path.read_bytes() for path in (out / "skymaps").iterdir()}
    (out / "summary.csv").unlink()
    with monkeypatch.context() as resume_patch:
        resume_patch.setattr("injstar.products.map_statistics", disabled)
        resume_patch.setattr("concurrent.futures.ProcessPoolExecutor", disabled)
        code, output = invoke("localize", localize_file, "--resume")
        assert code == 0, output
    assert read_summary(out) == first
    assert all(path.read_bytes() == data for path, data in kept_maps.items())

    source = config_file.parent / "out"
    before = {path.name: path.read_bytes() for path in source.iterdir()}
    second_file = config_file.parent / "second.toml"
    second_file.write_text(
        localize_file.read_text()
        .replace('"localized"', '"second"')
        .replace("5000.0", "6000.0")
    )
    code, output = invoke("localize", second_file)
    assert code == 0, output
    assert {path.name: path.read_bytes() for path in source.iterdir()} == before
    assert (out / "injections.h5").is_symlink()
    assert (out / "injections.h5").readlink() == Path("../out/injections.h5")
    assert (
        config_file.parent / "second" / "injections.h5"
    ).resolve() == source / "injections.h5"

    moved = config_file.parent / "moved"
    moved.mkdir()
    source.rename(moved / "out")
    out.rename(moved / "localized")
    (moved / "localized" / "summary.csv").unlink()
    build_summary(moved / "localized")
    assert read_summary(moved / "localized") == first


@pytest.mark.parametrize(
    "name", ["network_snr", "expected_snr_H1", "area_90", "localized"]
)
def test_generated_column_collision_rejected(
    config_file: Path, population_csv: Path, name: str
) -> None:
    lines = population_csv.read_text().splitlines()
    population_csv.write_text(
        "\n".join([lines[0] + "," + name, *(line + ",1" for line in lines[1:])]) + "\n"
    )
    code, output = invoke("inject", config_file)
    assert code == 1
    assert name in output


def test_partial_batches_are_stored_once(config_file: Path) -> None:
    code, output = invoke("inject", config_file)
    assert code == 0, output
    out = config_file.parent / "out"
    result = InjectionResult.read(out / "injection_results.h5")
    np.testing.assert_array_equal(result.row_index, [0, 1, 2])
    assert result.series.shape[0] == 3
    columns = read_population_catalogue(out / "injections.h5")
    np.testing.assert_array_equal(result.observed_snr[:, 0], columns["observed_snr_H1"])
    assert not list(out.glob("*.partial.h5"))


@pytest.mark.parametrize("missing", ["injection_results.h5", "injections.h5"])
def test_incomplete_simulation_is_recomputed(config_file: Path, missing: str) -> None:
    assert invoke("inject", config_file)[0] == 0
    out = config_file.parent / "out"
    (out / missing).unlink()
    (out / "snr.csv").write_text("legacy")
    code, output = invoke("inject", config_file)
    assert code == 0, output
    result = InjectionResult.read(out / "injection_results.h5")
    columns = read_population_catalogue(out / "injections.h5")
    np.testing.assert_array_equal(result.network_snr, columns["network_snr"])


def test_localize_without_detections(
    config_file: Path, population_csv: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lines = population_csv.read_text().splitlines()
    header = lines[0].split(",")
    distance = header.index("distance")
    for index in range(1, len(lines)):
        values = lines[index].split(",")
        values[distance] = "1000000000.0"
        lines[index] = ",".join(values)
    population_csv.write_text("\n".join(lines) + "\n")

    def disabled(*args: object, **kwargs: object) -> None:
        pytest.fail("no detections should not start localization")

    monkeypatch.setattr("concurrent.futures.ProcessPoolExecutor", disabled)
    assert invoke("inject", config_file)[0] == 0
    localize_file = make_localize_file(config_file)
    code, output = invoke("localize", localize_file)
    assert code == 0, output
    out = config_file.parent / "localized"
    assert read_summary(out) == []
    assert (out / "summary.csv").read_text().strip() == ",".join(
        ("row_index", *statistic_names())
    )
    assert not (out / "skymaps").exists()
    assert (
        InjectionResult.read(
            config_file.parent / "out" / "injection_results.h5"
        ).series.shape[0]
        == 3
    )

    source = config_file.parent / "out" / "injections.h5"
    written = source.stat().st_mtime_ns
    localize_file.write_text(localize_file.read_text().replace("5000.0", "6000.0"))
    assert invoke("localize", localize_file, "--resume")[0] == 1
    assert invoke("localize", localize_file, "--overwrite")[0] == 0
    assert source.stat().st_mtime_ns == written


def read_summary(output_dir: Path) -> list[dict[str, str]]:
    with (output_dir / "summary.csv").open(newline="") as stream:
        return list(csv.DictReader(stream))


def make_localize_file(config_file: Path) -> Path:
    path = config_file.parent / "localize.toml"
    path.write_text(
        'network = ["H1", "L1", "V1"]\n'
        'injection_dir = "out"\noutput_dir = "localized"\n'
        "[prior]\nmax_distance = 5000.0\n"
    )
    return path


@pytest.mark.parametrize(
    "missing", ["config.json", "injections.h5", "injection_results.h5"]
)
def test_localize_requires_completed_injection(config_file: Path, missing: str) -> None:
    assert invoke("inject", config_file)[0] == 0
    (config_file.parent / "out" / missing).unlink()
    code, output = invoke("localize", make_localize_file(config_file))
    assert code == 1
    assert f"no completed {missing}" in output
    assert not (config_file.parent / "localized").exists()


def test_localize_rejects_detector_order_before_overwrite(config_file: Path) -> None:
    assert invoke("inject", config_file)[0] == 0
    path = make_localize_file(config_file)
    path.write_text(
        path.read_text().replace('["H1", "L1", "V1"]', '["L1", "H1", "V1"]')
    )
    out = config_file.parent / "localized"
    out.mkdir()
    precious = out / "precious"
    precious.write_text("keep")
    code, output = invoke("localize", path, "--overwrite")
    assert code == 1
    assert "detector order" in output
    assert precious.read_text() == "keep"


def test_failed_injection_does_not_publish_partial_results(
    config_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from injstar.config import inject as config_inject

    original = config_inject._simulation_batches

    def interrupted(inputs: Inputs) -> Iterator[InjectionResult]:
        yield next(original(inputs))
        raise ValueError("interrupted injection")

    monkeypatch.setattr(config_inject, "_simulation_batches", interrupted)
    code, output = invoke("inject", config_file)
    assert code == 1
    assert "interrupted injection" in output
    out = config_file.parent / "out"
    assert sorted(path.name for path in out.iterdir()) == ["config.json"]
    monkeypatch.setattr(config_inject, "_simulation_batches", original)
    assert invoke("inject", config_file)[0] == 0


def test_localize_uses_saved_rows_flags_and_sampling(
    config_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from dataclasses import replace

    import h5py

    from injstar.dataset import ARRAY_FIELDS

    assert invoke("inject", config_file)[0] == 0
    source = config_file.parent / "out" / "injection_results.h5"
    result = InjectionResult.read(source)
    order = np.array([2, 0, 1])
    replace(
        result,
        **{name: np.asarray(getattr(result, name))[order] for name in ARRAY_FIELDS},
        sampling_frequency=8192.0,
    ).write(source)
    with h5py.File(source, "r+") as saved:
        saved["detected"][:] = [True, False, True]
    seen: list[tuple[Event, Network]] = []

    class Pool:
        def __init__(self, **kwargs: Any) -> None:
            pass

        def __enter__(self) -> "Pool":
            return self

        def __exit__(self, *args: Any) -> None:
            pass

        def map(
            self, task: Callable[[Event], int], batch: Iterable[Event]
        ) -> list[int]:
            return [task(event) for event in batch]

    def localize(event: Event, network: Network, **kwargs: Any) -> int:
        seen.append((event, network))
        return event.row_index

    monkeypatch.setattr("concurrent.futures.ProcessPoolExecutor", Pool)
    monkeypatch.setattr("injstar.localize.localize_to_file", localize)
    code, output = invoke("localize", make_localize_file(config_file))
    assert code == 0, output
    assert [event.row_index for event, _ in seen] == [2, 1]
    columns = read_population_catalogue(config_file.parent / "out" / "injections.h5")
    for position, (event, network) in zip([0, 2], seen, strict=True):
        for name in ("coa_time", "right_ascension", "declination", "distance"):
            assert getattr(event, name) == columns[name][position]
        assert network.sampling_frequency == 8192.0
        np.testing.assert_array_equal(event.series, result.series[event.row_index])


def test_noisy_chunks_match_full_injection(config_file: Path) -> None:
    from injstar.dataset import ARRAY_FIELDS

    config = load_inject_config(config_file)
    config.run()
    full = InjectionResult.read(config.output_dir / "injection_results.h5")
    chunks = []
    for start, stop, batch_size in [(0, 1, 1), (1, 3, 3)]:
        chunk = load_inject_config(
            config_file,
            start=start,
            stop=stop,
            batch_size=batch_size,
            output_dir=config_file.parent / f"chunk{start}",
        )
        chunk.run()
        chunks.append(InjectionResult.read(chunk.output_dir / "injection_results.h5"))
        catalogue = read_population_catalogue(chunk.output_dir / "injections.h5")
        truth = read_population_catalogue(config.output_dir / "injections.h5")
        for name in truth:
            np.testing.assert_allclose(catalogue[name], truth[name][start:stop])
    for name in ARRAY_FIELDS:
        np.testing.assert_allclose(
            np.concatenate([np.asarray(getattr(chunk, name)) for chunk in chunks]),
            getattr(full, name),
        )


def test_selection_overrides_and_legacy_reuse(config_file: Path) -> None:
    import json

    assert invoke("inject", config_file)[0] == 0
    out = config_file.parent / "out"
    target = out / "injections.h5"
    written = target.stat().st_mtime_ns
    snapshot = out / "config.json"
    old = json.loads(snapshot.read_text())
    del old["start"], old["stop"]
    snapshot.write_text(json.dumps(old))
    assert invoke("inject", config_file)[0] == 0
    assert target.stat().st_mtime_ns == written
    assert invoke("inject", config_file, "--start", 1)[0] == 1
    code, output = invoke(
        "inject",
        config_file,
        "--start",
        1,
        "--stop",
        100,
        "--output-dir",
        config_file.parent / "chunk",
    )
    assert code == 0, output
    result = InjectionResult.read(config_file.parent / "chunk" / "injection_results.h5")
    np.testing.assert_array_equal(result.row_index, [1, 2])
    recorded = json.loads((config_file.parent / "chunk" / "config.json").read_text())
    assert (recorded["start"], recorded["stop"]) == (1, 100)


@pytest.mark.parametrize(("start", "stop"), [(3, 4), (2, 1), (-1, 2), (0, -1)])
def test_invalid_selection_preserves_output(
    config_file: Path, start: int, stop: int
) -> None:
    assert invoke("inject", config_file)[0] == 0
    target = config_file.parent / "out" / "injections.h5"
    before = target.read_bytes()
    assert (
        invoke("inject", config_file, "--start", start, "--stop", stop, "--overwrite")[
            0
        ]
        != 0
    )
    assert target.read_bytes() == before


@pytest.mark.slow
def test_nonzero_chunk_localization_and_resume(config_file: Path) -> None:
    from dataclasses import asdict

    from ligo.skymap.io import read_sky_map

    from injstar.products import map_statistics

    chunk = config_file.parent / "chunk"
    assert (
        invoke("inject", config_file, "--start", 2, "--stop", 3, "--output-dir", chunk)[
            0
        ]
        == 0
    )
    localize_file = make_localize_file(config_file)
    code, output = invoke("localize", localize_file, "--injection-dir", chunk)
    assert code == 0, output
    out = config_file.parent / "localized"
    assert [path.name for path in (out / "skymaps").iterdir()] == ["2.multiorder.fits"]
    first = read_summary(out)
    assert [int(row["row_index"]) for row in first] == [2]
    truth = read_population_catalogue(chunk / "injections.h5")
    expected = map_statistics(
        read_sky_map(str(out / "skymaps" / "2.multiorder.fits"), moc=True, hdu=1),
        row_index=2,
        right_ascension=float(truth["right_ascension"][0]),
        declination=float(truth["declination"][0]),
        distance=float(truth["distance"][0]),
    )
    np.testing.assert_allclose(
        [float(value) for value in first[0].values()], list(asdict(expected).values())
    )
    written = (out / "skymaps" / "2.multiorder.fits").stat().st_mtime_ns
    code, output = invoke(
        "localize", localize_file, "--injection-dir", chunk, "--resume"
    )
    assert code == 0, output
    assert read_summary(out) == first
    assert (out / "skymaps" / "2.multiorder.fits").stat().st_mtime_ns == written


def test_collect_summary_cli(tmp_path: Path) -> None:
    from injstar.products import build_summary

    directories = [tmp_path / "one", tmp_path / "two"]
    for directory in directories:
        directory.mkdir()
        build_summary(directory)
    target = tmp_path / "combined.csv"
    code, output = invoke("collect-summary", *directories, "--output", target)
    assert code == 0, output
    assert target.read_bytes() == (directories[0] / "summary.csv").read_bytes()
    assert invoke("collect-summary", *directories, "--output", target)[0] == 1
    assert (
        invoke("collect-summary", *directories, "--output", target, "--overwrite")[0]
        == 0
    )
