"""Frequency-domain waveforms from PyCBC and LALSimulation."""

from collections.abc import Mapping
from dataclasses import dataclass
from functools import cached_property

import numpy as np
from numpy.typing import NDArray

from injstar.waveform import WaveformGenerator

_PARAMETERS = {
    "detector_frame_mass_1": "mass1",
    "detector_frame_mass_2": "mass2",
    "spin_1z": "spin1z",
    "spin_2z": "spin2z",
    "distance": "distance",
    "coa_phase": "coa_phase",
    "inclination": "inclination",
}
_IN_PLANE_SPINS = {
    "spin_1x": "spin1x",
    "spin_1y": "spin1y",
    "spin_2x": "spin2x",
    "spin_2y": "spin2y",
}


@dataclass(frozen=True)
class PyCBCWaveformGenerator(WaveformGenerator[np.ndarray]):
    """Generate LAL-backed PyCBC waveforms with the ``injstar[pycbc]`` extra."""

    approximant: str
    reference_frequency: float

    @cached_property
    def is_precessing(self) -> bool:
        """Whether LALSimulation classifies this model as precessing."""
        import lalsimulation

        approximant = lalsimulation.GetApproximantFromString(self.approximant)
        spin_support = lalsimulation.SimInspiralGetSpinSupportFromApproximant(
            approximant
        )
        if spin_support == lalsimulation.SIM_INSPIRAL_CASEBYCASE_SPINSUPPORT:
            raise ValueError(f"{self.approximant!r} has case-by-case spin support")
        return spin_support == lalsimulation.SIM_INSPIRAL_PRECESSINGSPIN

    @property
    def _parameters(self) -> Mapping[str, str]:
        return _PARAMETERS | _IN_PLANE_SPINS if self.is_precessing else _PARAMETERS

    @property
    def parameter_names(self) -> tuple[str, ...]:
        """Canonical population columns required by the model."""
        return tuple(self._parameters)

    def __call__(
        self, frequencies: np.ndarray, source: Mapping[str, np.ndarray | float]
    ) -> tuple[NDArray[np.complex128], NDArray[np.complex128]]:
        from pycbc.waveform import get_fd_waveform

        if frequencies.ndim != 1 or len(frequencies) < 2:
            raise ValueError("expected a one-dimensional grid of at least two bins")
        delta_f = float(frequencies[-1] - frequencies[0]) / (len(frequencies) - 1)
        if delta_f <= 0 or frequencies[0] <= 0 or not np.isfinite(frequencies).all():
            raise ValueError("expected finite, positive, increasing frequencies")
        bin_numbers = frequencies / delta_f
        bins = np.rint(bin_numbers).astype(np.intp)
        if not np.allclose(bin_numbers, bins, rtol=0, atol=1e-7) or not np.all(
            np.diff(bins) == 1
        ):
            raise ValueError("expected contiguous bins on a zero-origin FFT grid")

        parameters = {
            backend_name: float(source[name])
            for name, backend_name in self._parameters.items()
        }
        plus, cross = get_fd_waveform(
            approximant=self.approximant,
            delta_f=delta_f,
            f_lower=float(frequencies[0]),
            f_final=float(frequencies[-1] + delta_f),
            f_ref=self.reference_frequency,
            **parameters,
        )
        # Use the FD samples directly. The series epoch describes the FFT
        # segment and is not an extra coalescence-time shift.
        return (
            np.asarray(plus.numpy(), dtype=np.complex128)[bins],
            np.asarray(cross.numpy(), dtype=np.complex128)[bins],
        )
