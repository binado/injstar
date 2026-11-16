"""Frequency-domain waveforms from PyCBC and LALSimulation.

This module needs the optional ``injstar[pycbc]`` extra and raises an
:class:`ImportError` naming it when PyCBC or LALSimulation is missing.
:class:`PyCBCWaveformGenerator` implements the
:class:`~injstar.waveform.protocol.WaveformGenerator` protocol with NumPy arrays, so it
cannot run inside the jitted :func:`~injstar.inject.inject_events`. Use it to
cross-check the JAX backends against LAL.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from functools import cached_property

import numpy as np
from numpy.typing import NDArray

from injstar.waveform.protocol import WaveformGenerator

try:
    import lalsimulation
    from pycbc.waveform import get_fd_waveform
except ImportError as exc:
    raise ImportError(
        "injstar.waveform.pycbc needs the optional PyCBC dependencies; "
        "install them with `pip install 'injstar[pycbc]'`"
    ) from exc

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
_TIDES = {"lambda_1": "lambda1", "lambda_2": "lambda2"}


@dataclass(frozen=True)
class PyCBCWaveformGenerator(WaveformGenerator[np.ndarray]):
    """Generate LAL-backed PyCBC waveforms.

    Attributes
    ----------
    approximant : str
        LALSimulation approximant name, such as ``"IMRPhenomXAS"``. Models
        with case-by-case spin support are rejected when
        :attr:`is_precessing` is first read.
    reference_frequency : float
        Frequency in Hz at which the spin and orbital angles are defined.
    """

    approximant: str
    reference_frequency: float

    @cached_property
    def is_precessing(self) -> bool:
        """Whether LALSimulation classifies this model as precessing.

        Raises
        ------
        ValueError
            If LALSimulation reports case-by-case spin support, so the
            parameter set cannot be inferred from the approximant alone.
        """
        approximant = lalsimulation.GetApproximantFromString(self.approximant)
        spin_support = lalsimulation.SimInspiralGetSpinSupportFromApproximant(
            approximant
        )
        if spin_support == lalsimulation.SIM_INSPIRAL_CASEBYCASE_SPINSUPPORT:
            raise ValueError(f"{self.approximant!r} has case-by-case spin support")
        return spin_support == lalsimulation.SIM_INSPIRAL_PRECESSINGSPIN

    @cached_property
    def is_tidal(self) -> bool:
        """Whether the approximant name marks a model with tidal deformabilities.

        LALSimulation has no tidal-support classifier, so this matches the
        names that carry tides (``NRTidal*``, ``Lackey_Tidal_2013*`` and
        ``TaylorF2NLTides``).
        """
        return "Tidal" in self.approximant or "Tides" in self.approximant

    @property
    def _parameters(self) -> Mapping[str, str]:
        parameters = _PARAMETERS
        if self.is_precessing:
            parameters = parameters | _IN_PLANE_SPINS
        if self.is_tidal:
            parameters = parameters | _TIDES
        return parameters

    @property
    def parameter_names(self) -> tuple[str, ...]:
        """Canonical population columns required by the model.

        The aligned-spin set, plus the in-plane spin components when the model
        is precessing and the tidal deformabilities ``lambda_1`` and
        ``lambda_2`` when it is tidal.
        """
        return tuple(self._parameters)

    def __call__(
        self, frequencies: np.ndarray, source: Mapping[str, np.ndarray | float]
    ) -> tuple[NDArray[np.complex128], NDArray[np.complex128]]:
        """Return complex plus and cross strain on the supplied FFT bins.

        PyCBC generates the whole series from zero frequency up to
        ``frequencies[-1]`` and the requested bins are read out of it. The FD
        samples are used directly: the series epoch describes the FFT segment
        and is not an extra coalescence-time shift.

        Parameters
        ----------
        frequencies : numpy.ndarray
            Strictly increasing one-dimensional grid in Hz. It must be a run of
            consecutive integer multiples of ``delta_f``, so it lies on a FFT
            grid whose origin is zero. Strain below ``frequencies[0]`` is not
            generated.
        source : Mapping[str, numpy.ndarray or float]
            One source, keyed by the names in :attr:`parameter_names`. Each
            value must be a scalar. Masses are detector-frame solar masses,
            distance is in Mpc and angles are in radians.

        Returns
        -------
        plus, cross : numpy.ndarray
            ``complex128`` strain with the shape of ``frequencies``, using the
            LAL convention. The arrays are not JAX arrays, so they cannot be
            traced or differentiated.

        Raises
        ------
        ValueError
            If ``frequencies`` is not one-dimensional with at least two bins,
            is not finite, positive and increasing, or is not a contiguous
            run of bins on a zero-origin FFT grid.
        """
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
            # f_final is exclusive, so extend it by one bin to keep the last one.
            f_final=float(frequencies[-1] + delta_f),
            f_ref=self.reference_frequency,
            **parameters,
        )
        return (
            np.asarray(plus.numpy(), dtype=np.complex128)[bins],
            np.asarray(cross.numpy(), dtype=np.complex128)[bins],
        )
