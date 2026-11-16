from pathlib import Path

import pytest
from pydantic import ValidationError

from injstar.config import (
    InjectConfig,
    LocalizeConfig,
    PriorConfig,
    load_inject_config,
    load_localize_config,
)

from .conftest import make_config


def test_paths_resolve_against_the_config_file(config_file: Path) -> None:
    config = load_inject_config(config_file)

    assert config.population == config_file.parent / "population.csv"
    assert config.output_dir == config_file.parent / "out"
    assert config.network == ("H1", "L1", "V1")
    assert config.segment_samples == 16384
    assert config.crop_samples % 2 == 1


def test_overrides_replace_file_values(config_file: Path, tmp_path: Path) -> None:
    config = load_inject_config(
        config_file, output_dir=tmp_path / "elsewhere", seed=7, batch_size=None
    )

    assert config.output_dir == (tmp_path / "elsewhere").resolve()
    assert config.seed == 7
    assert config.batch_size == 2


def test_json_round_trip(config_file: Path) -> None:
    config = load_inject_config(config_file)

    assert InjectConfig.model_validate_json(config.model_dump_json()) == config


def test_json_round_trip_keeps_the_template_approximant(tmp_path: Path) -> None:
    config = make_config(
        tmp_path,
        waveform_approximant="IMRPhenomXPHM",
        template_approximant="IMRPhenomXAS",
    )

    restored = InjectConfig.model_validate_json(config.model_dump_json())

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
        ({"unknown_field": 1}, "Extra inputs"),
    ],
)
def test_invalid_configs_are_rejected(
    tmp_path: Path, fields: dict[str, object], message: str
) -> None:
    with pytest.raises(ValidationError, match=message):
        make_config(tmp_path, **fields)


def test_localize_prior_is_required(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="prior"):
        LocalizeConfig.model_validate(
            {
                "network": "x",
                "injection_dir": tmp_path / "in",
                "output_dir": tmp_path / "out",
            }
        )


@pytest.mark.parametrize(
    "fields",
    [
        {"max_distance": 10.0, "min_distance": 20.0},
        {"max_distance": 10.0, "cosmology": True, "distance_power": 1},
        {"max_distance": 10.0, "distance_power": -1},
    ],
)
def test_invalid_prior(fields: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        PriorConfig.model_validate(fields)


def test_localize_paths_overrides_and_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "localize.toml"
    path.write_text(
        'network = "HLV"\ninjection_dir = "in"\noutput_dir = "out"\n'
        'detector_config = ["detectors.toml"]\n[prior]\nmax_distance = 5000.0\n'
    )
    config = load_localize_config(path)
    assert config.injection_dir == tmp_path / "in"
    assert config.detector_config == (tmp_path / "detectors.toml",)
    assert LocalizeConfig.model_validate_json(config.model_dump_json()) == config
    monkeypatch.chdir(tmp_path)
    replaced = load_localize_config(path, output_dir="elsewhere", workers=3)
    assert replaced.output_dir == tmp_path / "elsewhere"
    assert replaced.workers == 3


@pytest.mark.parametrize("target", ["in", "in/child", "."])
def test_overlapping_directories_rejected(tmp_path: Path, target: str) -> None:
    with pytest.raises(ValidationError, match="overlap"):
        LocalizeConfig(
            network="HLV",
            injection_dir=tmp_path / "in",
            output_dir=tmp_path / target,
            prior=PriorConfig(max_distance=5000.0),
        )
