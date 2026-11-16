r"""Frequency-domain waveform models.

Submodules provide backends (:mod:`injstar.waveform.ripple` for ripple,
:mod:`injstar.waveform.pycbc` for PyCBC/LALSimulation); this package exposes
the :class:`WaveformGenerator` protocol they implement and re-exports the
ripple backend as the default.
"""

from injstar.waveform.protocol import WaveformGenerator
from injstar.waveform.ripple import RippleWaveformGenerator, waveform

__all__ = ["RippleWaveformGenerator", "WaveformGenerator", "waveform"]
