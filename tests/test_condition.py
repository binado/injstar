"""Compare injstar's detector inputs with ``ligo.skymap.bayestar.condition``.

The same zero-noise SNR series is handed to ``condition``. The reference PSD
restricts its conditioned LAL template to the injection band: conditioning
otherwise adds power below the requested low-frequency cutoff. The horizon
comparison accounts for trapezoidal versus rectangular endpoint weighting.
"""

from dataclasses import dataclass

import lal
import numpy as np
import pytest
from ligo.skymap import bayestar
from ligo.skymap.bayestar import filter
from ligo.skymap.io import events
from numpy.typing import NDArray

from injstar.detector.config import DetectorRegistry
from injstar.localize import Event, Network

from .tolerances import assert_close

PSD_DELTA_F = 0.25
PSD_F_MAX = 2048.0
MINIMUM_FREQUENCY = 20.0
MAXIMUM_FREQUENCY = 1024.0


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
    in_band = (frequencies >= MINIMUM_FREQUENCY) & (frequencies <= MAXIMUM_FREQUENCY)
    series.data.data = np.where(in_band & np.isfinite(values), values, np.inf)
    return series


@dataclass(frozen=True)
class Reference:
    """Output of ``bayestar.condition`` and its squared-horizon correction."""

    epoch: lal.LIGOTimeGPS
    sampling_frequency: float
    toas: NDArray[np.float64]
    snrs: NDArray[np.float64]
    responses: NDArray[np.float64]
    locations: NDArray[np.float64]
    horizons: NDArray[np.float64]
    horizon_endpoint_corrections: NDArray[np.float64]


@pytest.fixture
def reference(
    event_and_network: tuple[Event, Network], registry: DetectorRegistry
) -> Reference:
    event, network = event_and_network
    psds = [_psd(registry, name) for name in network.names]
    singles = [
        _Single(name, event, index, psd, network.sampling_frequency)
        for index, (name, psd) in enumerate(zip(network.names, psds))
    ]
    template = filter.sngl_inspiral_psd(
        "IMRPhenomD", f_min=MINIMUM_FREQUENCY, **_Event.template_args
    )
    endpoint_corrections = []
    for psd in psds:
        model = filter.SignalModel(
            filter.signal_psd_series(
                template,
                filter.InterpolatedPSD(filter.abscissa(psd), psd.data.data),
            )
        )
        # SignalModel trims zero-weight bins, then gives each nonzero endpoint
        # half weight. Restore those halves to match injstar's rectangle sum.
        endpoint_corrections.append(
            0.5 * model.dw * (model.denom_integrand[0] + model.denom_integrand[-1])
        )
    return Reference(
        *bayestar.condition(
            _Event(singles),
            waveform="IMRPhenomD",
            f_low=MINIMUM_FREQUENCY,
            f_high_truncate=1.0,
        ),
        horizon_endpoint_corrections=np.asarray(endpoint_corrections),
    )


def test_geodetic_responses_match_lal_tensors(
    reference: Reference, event_and_network: tuple[Event, Network]
) -> None:
    assert_close(
        event_and_network[1].responses, reference.responses, promote_dtypes=True
    )


def test_horizons_match_lal_template_with_endpoint_correction(
    reference: Reference, event_and_network: tuple[Event, Network]
) -> None:
    rectangle_horizons = np.sqrt(
        reference.horizons**2 + reference.horizon_endpoint_corrections
    )
    assert_close(event_and_network[0].horizon, rectangle_horizons)
