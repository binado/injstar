import importlib
import subprocess
import sys

import pytest

MODULES = [
    "cli",
    "config",
    "constants",
    "localize",
    "detector",
    "detector.geometry",
    "detector.noise",
    "detector.projection",
    "detector.psd",
    "config.inject",
    "config.localize",
    "population",
    "io",
    "products",
    "runtime",
    "inject",
    "waveform",
]


@pytest.mark.parametrize("name", MODULES)
def test_module_imports(name: str) -> None:
    module = importlib.import_module(f"injstar.{name}")

    assert module.__doc__


@pytest.mark.parametrize(
    ("name", "forbidden"),
    [
        ("cli", {"astropy", "jax", "ligo.skymap", "pydantic"}),
        ("config", {"astropy", "jax", "ligo.skymap"}),
        ("detector", {"astropy", "jax", "ligo.skymap"}),
        ("detector.geometry", {"astropy", "jax", "ligo.skymap"}),
        ("detector.psd", {"astropy", "jax", "ligo.skymap"}),
        ("localize", {"jax"}),
        ("products", {"astropy", "jax", "ligo.skymap"}),
    ],
)
def test_import_stays_light(name: str, forbidden: set[str]) -> None:
    code = (
        f"import sys, injstar.{name}; "
        f"heavy = {forbidden!r} & set(sys.modules); "
        "sys.exit(1 if heavy else 0)"
    )

    assert subprocess.run([sys.executable, "-c", code], check=False).returncode == 0
