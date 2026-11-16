"""Protocol for frequency-domain waveform models."""

from collections.abc import Mapping
from typing import Protocol


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
