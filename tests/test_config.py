from pathlib import Path

import pytest
from pydantic import ValidationError

from injstar.config import RunConfig, load_run_config

from .conftest import make_config


def test_paths_resolve_against_the_config_file(config_file: Path) -> None:
    config = load_run_config(config_file)

    assert config.population == config_file.parent / "population.csv"
    assert config.output_dir == config_file.parent / "out"
    assert config.network == ("H1", "L1", "V1")
    assert config.segment_samples == 16384
    assert config.crop_samples % 2 == 1


def test_overrides_replace_file_values(config_file: Path, tmp_path: Path) -> None:
    config = load_run_config(
        config_file, output_dir=tmp_path / "elsewhere", seed=7, workers=None
    )

    assert config.output_dir == (tmp_path / "elsewhere").resolve()
    assert config.seed == 7
    assert config.workers == 1


def test_json_round_trip(config_file: Path) -> None:
    config = load_run_config(config_file)

    assert RunConfig.model_validate_json(config.model_dump_json()) == config


def test_json_round_trip_keeps_the_template_approximant(tmp_path: Path) -> None:
    config = make_config(
        tmp_path,
        waveform_approximant="IMRPhenomXPHM",
        template_approximant="IMRPhenomXAS",
    )

    restored = RunConfig.model_validate_json(config.model_dump_json())

    assert restored == config
    assert restored.template_approximant == "IMRPhenomXAS"


def test_template_approximant_defaults_to_the_signal_model(tmp_path: Path) -> None:
    assert make_config(tmp_path).template_approximant is None


@pytest.mark.parametrize(
    ("fields", "message"),
    [
        ({"maximum_frequency": 4096.0}, "sampling_frequency / 2"),
        (
            {"minimum_frequency": 2000.0},
            "maximum_frequency must exceed minimum_frequency",
        ),
        ({"segment_duration": 4.0001}, "whole number of samples"),
        ({"crop_duration": 3.0}, "under half"),
        ({"prior": {"max_distance": 10.0, "min_distance": 20.0}}, "max_distance"),
        (
            {"prior": {"max_distance": 10.0, "cosmology": True, "distance_power": 1}},
            "distance_power = 2",
        ),
        ({"unknown_field": 1}, "Extra inputs"),
    ],
)
def test_invalid_configs_are_rejected(
    tmp_path: Path, fields: dict[str, object], message: str
) -> None:
    with pytest.raises(ValidationError, match=message):
        make_config(tmp_path, **fields)


def test_max_distance_is_required(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="prior"):
        RunConfig.model_validate(
            {"network": "x", "population": tmp_path, "output_dir": tmp_path}
        )
