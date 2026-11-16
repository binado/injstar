"""Shared path loading and output-directory safety."""

import json
import shutil
import tomllib
from pathlib import Path
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field

from injstar import io

_STRICT = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)
Positive = Annotated[float, Field(gt=0)]
_EXECUTION_FIELDS = frozenset(
    {"output_dir", "workers", "threads_per_worker", "batch_size"}
)


class RunDirectoryError(RuntimeError):
    """The output directory cannot be used for the requested stage."""


def load_config[T: BaseModel](path: Path, model: type[T], **overrides: Any) -> T:
    """Resolve TOML paths beside the file and override paths against cwd."""
    path = Path(path)
    with path.open("rb") as handle:
        data = tomllib.load(handle)
    base = path.resolve().parent
    paths = {"population", "output_dir", "injection_dir"}
    for key in paths & data.keys():
        data[key] = str(base / data[key])
    if "detector_config" in data:
        data["detector_config"] = [str(base / item) for item in data["detector_config"]]
    for key, value in overrides.items():
        if value is not None:
            if key in paths:
                value = Path(value).resolve()
            elif key == "detector_config":
                value = tuple(Path(item).resolve() for item in value)
            data[key] = value
    return model.model_validate(data)


def _results_key(config: dict[str, Any]) -> dict[str, Any]:
    """Fields that determine scientific results."""
    if "population" in config:
        config = {"start": 0, "stop": None, **config}
    return {k: v for k, v in config.items() if k not in _EXECUTION_FIELDS}


def prepare_directory(config: Any, *, overwrite: bool) -> bool:
    """Prepare an output directory, refusing deletion of foreign contents."""
    output_dir = config.output_dir
    snapshot = output_dir / io.CONFIG_FILE
    occupied = output_dir.exists() and any(output_dir.iterdir())
    recorded = json.loads(snapshot.read_text()) if snapshot.is_file() else None
    requested = config.model_dump(mode="json")
    if recorded is not None and _results_key(recorded) == _results_key(requested):
        if not overwrite:
            return True
    elif occupied and not overwrite:
        reason = (
            "holds a run with a different config"
            if recorded is not None
            else "is not empty and is not an injstar run"
        )
        raise RunDirectoryError(
            f"{output_dir} {reason}; pass --overwrite to replace it"
        )
    if occupied:
        if recorded is None:
            raise RunDirectoryError(
                f"{output_dir} is not an injstar run directory; refusing to delete it"
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return False


def write_config(config: Any) -> None:
    """Record the config in the run directory."""
    (config.output_dir / io.CONFIG_FILE).write_text(
        config.model_dump_json(indent=2) + "\n"
    )
