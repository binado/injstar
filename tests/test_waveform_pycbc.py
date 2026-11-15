import subprocess
import sys

import numpy as np
import pytest

pytest.importorskip("pycbc")

from injstar.waveform.pycbc import PyCBCWaveformGenerator  # noqa: E402

MASSES = ("detector_frame_mass_1", "detector_frame_mass_2")
ALIGNED = (*MASSES, "spin_1z", "spin_2z", "distance", "coa_phase", "inclination")
IN_PLANE = ("spin_1x", "spin_1y", "spin_2x", "spin_2y")
TIDES = ("lambda_1", "lambda_2")


@pytest.fixture
def source() -> dict[str, float]:
    return {
        "detector_frame_mass_1": 1.5,
        "detector_frame_mass_2": 1.3,
        "spin_1z": 0.02,
        "spin_2z": -0.01,
        "distance": 100.0,
        "coa_phase": 0.0,
        "inclination": 0.3,
        "lambda_1": 400.0,
        "lambda_2": 500.0,
    }


@pytest.fixture
def frequencies() -> np.ndarray:
    return np.arange(80, 2049) * 0.25


@pytest.mark.parametrize(
    ("approximant", "names"),
    [
        ("IMRPhenomD", ALIGNED),
        ("IMRPhenomXPHM", (*ALIGNED, *IN_PLANE)),
        ("IMRPhenomD_NRTidalv2", (*ALIGNED, *TIDES)),
        ("IMRPhenomPv2_NRTidalv2", (*ALIGNED, *IN_PLANE, *TIDES)),
    ],
)
def test_parameter_names_follow_the_model_physics(
    approximant: str, names: tuple[str, ...]
) -> None:
    model = PyCBCWaveformGenerator(approximant, reference_frequency=20.0)

    assert model.parameter_names == names


@pytest.mark.parametrize(
    ("approximant", "is_precessing"),
    [("IMRPhenomD", False), ("IMRPhenomXPHM", True), ("IMRPhenomD_NRTidalv2", False)],
)
def test_is_precessing_follows_lal_spin_support(
    approximant: str, is_precessing: bool
) -> None:
    model = PyCBCWaveformGenerator(approximant, reference_frequency=20.0)

    assert model.is_precessing is is_precessing


@pytest.mark.parametrize(
    ("approximant", "is_tidal"),
    [("IMRPhenomD", False), ("IMRPhenomXPHM", False), ("IMRPhenomD_NRTidalv2", True)],
)
def test_is_tidal_follows_the_approximant_name(
    approximant: str, is_tidal: bool
) -> None:
    model = PyCBCWaveformGenerator(approximant, reference_frequency=20.0)

    assert model.is_tidal is is_tidal


def test_call_tidal_model_depends_on_the_deformabilities(
    frequencies: np.ndarray, source: dict[str, float]
) -> None:
    model = PyCBCWaveformGenerator("IMRPhenomD_NRTidalv2", reference_frequency=20.0)

    plus, _ = model(frequencies, source)
    stiffer, _ = model(frequencies, source | {"lambda_1": 800.0, "lambda_2": 900.0})

    assert plus.shape == frequencies.shape
    # Strain is ~1e-18, so only a purely relative comparison can tell them apart.
    assert not np.allclose(plus, stiffer, rtol=1e-3, atol=0)


def test_import_without_pycbc_names_the_extra_in_the_error() -> None:
    code = "import sys; sys.modules['pycbc'] = None; import injstar.waveform.pycbc"

    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )

    assert result.returncode != 0
    assert "injstar[pycbc]" in result.stderr
