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
    "detector.projection",
    "detector.psd",
    "pipeline",
    "population",
    "products",
    "runtime",
    "simulate",
]


@pytest.mark.parametrize("name", MODULES)
def test_module_imports(name: str) -> None:
    module = importlib.import_module(f"injstar.{name}")

    assert module.__doc__


def test_cli_import_stays_light() -> None:
    code = (
        "import sys, injstar.cli; "
        "heavy = {'jax', 'ligo.skymap', 'astropy'} & set(sys.modules); "
        "sys.exit(1 if heavy else 0)"
    )

    assert subprocess.run([sys.executable, "-c", code], check=False).returncode == 0
