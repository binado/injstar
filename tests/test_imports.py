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
    "detector.network",
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


@pytest.mark.parametrize("name", ["cli", "detector"])
def test_import_stays_light(name: str) -> None:
    code = (
        f"import sys, injstar.{name}; "
        "heavy = {'jax', 'ligo.skymap', 'astropy'} & set(sys.modules); "
        "sys.exit(1 if heavy else 0)"
    )

    assert subprocess.run([sys.executable, "-c", code], check=False).returncode == 0
