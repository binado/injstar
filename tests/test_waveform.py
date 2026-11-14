import jax.numpy as jnp
import numpy as np
import pytest
import ripplegw

from injstar.population import CANONICAL_COLUMNS
from injstar.waveform import Waveform, waveform

MASSES = ("detector_frame_mass_1", "detector_frame_mass_2")
TAIL = ("distance", "coa_phase", "inclination")


@pytest.fixture
def source() -> dict[str, float]:
    return {
        "detector_frame_mass_1": 36.0,
        "detector_frame_mass_2": 29.0,
        "spin_1z": 0.1,
        "spin_2z": -0.1,
        "distance": 1.0,
        "coa_phase": 0.0,
        "inclination": 0.0,
    }


def test_waveform_unknown_approximant_raises() -> None:
    with pytest.raises(ValueError, match="IMRPhenomD"):
        waveform("NotAModel", reference_frequency=20.0)


@pytest.mark.parametrize(
    ("approximant", "names", "is_precessing", "highest_mode"),
    [
        ("IMRPhenomD", (*MASSES, "spin_1z", "spin_2z", *TAIL), False, 2),
        (
            "IMRPhenomXPHM",
            (
                *MASSES,
                "spin_1x",
                "spin_1y",
                "spin_1z",
                "spin_2x",
                "spin_2y",
                "spin_2z",
                *TAIL,
            ),  # fmt: skip
            True,
            4,
        ),
        (
            "TaylorF2",
            (*MASSES, "spin_1z", "spin_2z", "lambda_1", "lambda_2", *TAIL),
            False,
            2,
        ),
    ],
)
def test_waveform_reports_the_model_parameters(
    approximant: str,
    names: tuple[str, ...],
    is_precessing: bool,
    highest_mode: int,
) -> None:
    model = waveform(approximant, reference_frequency=20.0)

    assert isinstance(model, Waveform)
    assert model.parameter_names == names
    assert model.is_precessing is is_precessing
    assert model.highest_mode == highest_mode


def test_polarizations_face_on_cross_is_minus_i_plus(
    source: dict[str, float],
) -> None:
    model = waveform("IMRPhenomD", reference_frequency=20.0)
    frequencies = jnp.linspace(20.0, 512.0, 50)

    plus, cross = model.polarizations(frequencies, source)

    np.testing.assert_allclose(cross, -1j * plus, rtol=1e-12, atol=0)
    assert np.all(np.abs(plus) > 0)


def test_polarizations_zero_samples_outside_the_model_band(
    source: dict[str, float],
) -> None:
    model = waveform("IMRPhenomD", reference_frequency=20.0)
    frequencies = jnp.array([20.0, 1e6])

    plus, cross = model.polarizations(frequencies, source)

    assert np.all(np.isfinite(plus)) and np.all(np.isfinite(cross))


def test_polarizations_missing_parameter_raises(source: dict[str, float]) -> None:
    model = waveform("IMRPhenomD", reference_frequency=20.0)
    del source["spin_2z"]

    with pytest.raises(KeyError, match="spin_2z"):
        model.polarizations(jnp.array([30.0]), source)


@pytest.mark.parametrize(
    "approximant", ripplegw.list_waveforms(domain="FD", source_type="cbc")
)
def test_waveform_parameters_are_population_columns(approximant: str) -> None:
    model = waveform(approximant, reference_frequency=20.0)

    assert set(model.parameter_names) <= set(CANONICAL_COLUMNS)
