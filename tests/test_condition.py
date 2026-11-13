"""Compare injstar's detector inputs with ``ligo.skymap.bayestar.condition``.

The same zero-noise SNR series is handed to ``condition``. The outputs differ
only where the inputs differ: injstar's response tensors are geodetic where
LAL's are tabulated, and its horizon comes from ripplegw's IMRPhenomD where
``condition`` integrates LAL's.
"""

from dataclasses import dataclass

import lal
import numpy as np
import pytest
from ligo.skymap import bayestar
from ligo.skymap.io import events
from numpy.typing import NDArray

from injstar.detector.network import DetectorRegistry
from injstar.localize import Event, Network

PSD_DELTA_F = 0.25
PSD_F_MAX = 2048.0


class _Single(events.SingleEvent):
    def __init__(
        self,
        name: str,
        event: Event,
        index: int,
        psd: lal.REAL8FrequencySeries,
        rate: float,
    ) -> None:
        samples = event.series.shape[1]
        start = event.coa_time + event.epochs[index]
        seconds = int(np.floor(start))
        series = lal.CreateCOMPLEX8TimeSeries(
            "snr", lal.LIGOTimeGPS(0), 0, 1 / rate, lal.StrainUnit, samples
        )
        series.epoch = lal.LIGOTimeGPS(seconds, int(round(1e9 * (start - seconds))))
        series.data.data = event.series[index].astype(np.complex64)
        self._name = name
        self._snr = float(event.snr[index])
        self._phase = float(np.angle(event.series[index, samples // 2]))
        self._time = float(series.epoch) + 0.5 * (samples - 1) / rate
        self._psd = psd
        self._series = series

    detector = property(lambda self: self._name)
    snr = property(lambda self: self._snr)
    phase = property(lambda self: self._phase)
    time = property(lambda self: self._time)
    zerolag_time = time
    psd = property(lambda self: self._psd)
    snr_series = property(lambda self: self._series)


class _Event(events.Event):
    def __init__(self, singles: list[_Single]) -> None:
        self._singles = singles

    singles = property(lambda self: self._singles)
    template_args = {"mass1": 36.0, "mass2": 29.0, "spin1z": 0.0, "spin2z": 0.0}


def _psd(registry: DetectorRegistry, name: str) -> lal.REAL8FrequencySeries:
    frequencies = np.arange(round(PSD_F_MAX / PSD_DELTA_F) + 1) * PSD_DELTA_F
    values = registry.detectors[name].sensitivity(frequencies)
    series = lal.CreateREAL8FrequencySeries(
        "psd", 0, 0.0, PSD_DELTA_F, lal.DimensionlessUnit, len(frequencies)
    )
    series.data.data = np.where(np.isfinite(values) & (frequencies > 0), values, np.inf)
    return series


@dataclass(frozen=True)
class Reference:
    """Output of ``bayestar.condition``."""

    epoch: lal.LIGOTimeGPS
    sample_rate: float
    toas: NDArray[np.float64]
    snrs: NDArray[np.float64]
    responses: NDArray[np.float64]
    locations: NDArray[np.float64]
    horizons: NDArray[np.float64]


@pytest.fixture
def reference(
    event_and_network: tuple[Event, Network], registry: DetectorRegistry
) -> Reference:
    event, network = event_and_network
    singles = [
        _Single(name, event, index, _psd(registry, name), network.sample_rate)
        for index, name in enumerate(network.names)
    ]
    return Reference(
        *bayestar.condition(
            _Event(singles), waveform="IMRPhenomD", f_low=20.0, f_high_truncate=1.0
        )
    )


def test_geodetic_responses_match_lal_tensors_to_1e_3(
    reference: Reference, event_and_network: tuple[Event, Network]
) -> None:
    # The audit records up to 3e-4 for H1 and L1 and 2e-3 for K1 (not in HLV).
    np.testing.assert_allclose(
        event_and_network[1].responses, reference.responses, atol=1e-3
    )


def test_horizons_match_lal_template_within_half_a_percent(
    reference: Reference, event_and_network: tuple[Event, Network]
) -> None:
    np.testing.assert_allclose(
        event_and_network[0].horizon, reference.horizons, rtol=5e-3
    )
