r"""Frequency-domain waveform models.

A thin wrapper around :mod:`ripplegw`, the only module that imports it. It
translates between the canonical population column names (``distance``,
``coa_phase``, ``spin_1x``, ...) and the names ripple's models read (``d_L``,
``phase_c``, ``s1_x``, ...), so the rest of the package neither knows nor
assumes which parameters a model needs: whether it is aligned-spin, precessing
or tidal is read from the model.

:meth:`RippleWaveformGenerator.__call__` evaluates one source and returns the complex
plus and cross polarizations at the given frequencies. For the dominant mode
they obey :math:`h_\times = -i \, 2\cos\iota / (1 + \cos^2\iota) \, h_+`;
face-on this is :math:`h_\times = -i h_+`.

JAX must run in 64-bit mode: call :func:`injstar.runtime.configure_runtime`
first.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Protocol

import jax
import jax.numpy as jnp
import ripplegw
import ripplegw.conversions
from numpy.typing import ArrayLike


class WaveformGenerator[ArrayT](Protocol):
    """One configured frequency-domain compact-binary model."""

    @property
    def parameter_names(self) -> tuple[str, ...]:
        """Canonical population columns required by the model."""
        ...

    def __call__(
        self, frequencies: ArrayT, source: Mapping[str, ArrayT | float]
    ) -> tuple[ArrayT, ArrayT]:
        """Return complex plus and cross strain on the supplied FFT bins.

        Frequencies are in Hz, masses are detector-frame solar masses,
        distance is in Mpc, and angles are in radians. The polarizations use
        the LAL convention with no detector projection or arrival-time shift.
        """
        ...


_RIPPLE_NAMES = {
    "spin_1x": "s1_x",
    "spin_1y": "s1_y",
    "spin_1z": "s1_z",
    "spin_2x": "s2_x",
    "spin_2y": "s2_y",
    "spin_2z": "s2_z",
    "lambda_1": "lambda_1",
    "lambda_2": "lambda_2",
    "distance": "d_L",
    "coa_phase": "phase_c",
    "inclination": "iota",
}
_MASSES = ("detector_frame_mass_1", "detector_frame_mass_2")
_MASS_PARAMETERS = ("M_c", "eta")
_CANONICAL_NAMES = {ripple: name for name, ripple in _RIPPLE_NAMES.items()}


@dataclass(frozen=True)
class RippleWaveformGenerator:
    """A frequency-domain compact-binary waveform model.

    Construct directly or build it with :func:`waveform`.

    Attributes
    ----------
    approximant
        Name of the model.
    reference_frequency
        Reference frequency of the spins and phase in Hz.
    parameter_names
        Canonical population columns the model reads.
    is_precessing
        Whether the model has precessing spins.
    highest_mode
        Highest multipole ``l`` the model includes; 2 for dominant-mode models.
    """

    approximant: str
    reference_frequency: float
    _model: ripplegw.Waveform = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Build the backend and validate its population parameters."""
        choices = ripplegw.list_waveforms(domain="FD", source_type="cbc")
        if self.approximant not in choices:
            raise ValueError(
                f"unknown waveform approximant {self.approximant!r}; choose from {choices}"
            )
        model = ripplegw.waveform(self.approximant, f_ref=self.reference_frequency)
        _canonical_names(tuple(model.parameter_names))
        object.__setattr__(self, "_model", model)

    @property
    def parameter_names(self) -> tuple[str, ...]:
        """Canonical population columns the backend model reads."""
        return _canonical_names(tuple(self._model.parameter_names))

    @property
    def is_precessing(self) -> bool:
        """Whether the backend model has precessing spins."""
        return bool(self._model.waveform_metadata["is_precessing"])

    @property
    def highest_mode(self) -> int:
        """Highest multipole ``l`` included by the backend model."""
        return int(self._model.waveform_metadata["highest_mode"])

    def __call__(
        self, frequencies: jax.Array, source: Mapping[str, ArrayLike]
    ) -> tuple[jax.Array, jax.Array]:
        """Plus and cross polarizations of one source.

        Parameters
        ----------
        frequencies
            Frequencies in Hz, shape ``(bins,)``.
        source
            Scalar value of each name in :attr:`parameter_names`: masses in
            detector-frame solar masses, ``distance`` in Mpc, angles in
            radians. Other entries are ignored.

        Returns
        -------
        plus, cross
            Complex strain, shape ``(bins,)`` each, with non-finite samples set
            to zero.

        Raises
        ------
        KeyError
            If ``source`` lacks a parameter the model reads.
        """
        names = self.parameter_names
        missing = [name for name in names if name not in source]
        if missing:
            raise KeyError(f"{self.approximant} needs the parameters {missing}")
        params: dict[str, jax.Array] = {}
        if all(name in names for name in _MASSES):
            chirp_mass, eta = ripplegw.conversions.ms_to_Mc_eta(
                jnp.stack([jnp.asarray(source[name]) for name in _MASSES])
            )
            params = {"M_c": chirp_mass, "eta": eta}
        for name in names:
            if name in _RIPPLE_NAMES:
                params[_RIPPLE_NAMES[name]] = jnp.asarray(source[name])
        strain = self._model(frequencies, params)
        return _finite(strain["p"]), _finite(strain["c"])


def _finite(values: jax.Array) -> jax.Array:
    """``values`` with non-finite samples set to zero."""
    return jnp.where(jnp.isfinite(values), values, 0.0)


def _canonical_names(ripple_names: tuple[str, ...]) -> tuple[str, ...]:
    """Population columns behind a model's own parameter names."""
    names: list[str] = []
    for ripple in ripple_names:
        if ripple in _MASS_PARAMETERS:
            added: tuple[str, ...] = _MASSES
        elif ripple in _CANONICAL_NAMES:
            added = (_CANONICAL_NAMES[ripple],)
        else:
            raise ValueError(f"waveform parameter {ripple!r} has no population column")
        names.extend(name for name in added if name not in names)
    return tuple(names)


def waveform(
    approximant: str, *, reference_frequency: float
) -> RippleWaveformGenerator:
    """Build a frequency-domain waveform model.

    Parameters
    ----------
    approximant
        Name of a ripple compact-binary frequency-domain model, for example
        ``"IMRPhenomD"``.
    reference_frequency
        Reference frequency of the spins and phase in Hz.

    Raises
    ------
    ValueError
        If the approximant is unknown, or the model reads a parameter with no
        population column.
    """
    return RippleWaveformGenerator(
        approximant=approximant,
        reference_frequency=reference_frequency,
    )
