"""Compare :func:`injstar.localize.pack` with ``ligo.skymap.bayestar.condition``.

The same zero-noise SNR series is handed to both. They differ only where the
inputs differ: injstar's response tensors are geodetic where LAL's are
tabulated, and its horizon and bandwidth come from ripplegw's IMRPhenomD where
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
from injstar.localize import Event, Network, Packed, pack

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


@pytest.fixture
def packed(event_and_network: tuple[Event, Network]) -> Packed:
    return pack(*event_and_network)


def test_pack_gives_bayestar_sample_rate_and_shape(
    packed: Packed, reference: Reference, event_and_network: tuple[Event, Network]
) -> None:
    assert event_and_network[1].sample_rate == reference.sample_rate
    assert packed.snrs.shape == reference.snrs.shape


def test_pack_gives_bayestar_epoch_and_gmst(
    packed: Packed, reference: Reference
) -> None:
    # The epochs differ through the timing weights, which use independent
    # bandwidths; the reference also averages in integer nanoseconds.
    assert float(reference.epoch - packed.epoch) == pytest.approx(0.0, abs=1e-6)
    assert packed.gmst == pytest.approx(
        lal.GreenwichMeanSiderealTime(reference.epoch), abs=1e-9
    )


def test_pack_gives_bayestar_arrival_times_to_a_fraction_of_a_sample(
    packed: Packed, reference: Reference, event_and_network: tuple[Event, Network]
) -> None:
    rate = event_and_network[1].sample_rate
    np.testing.assert_allclose((packed.epochs - reference.toas) * rate, 0.0, atol=2e-3)


def test_pack_gives_bayestar_light_travel_times(
    packed: Packed, reference: Reference
) -> None:
    np.testing.assert_allclose(packed.locations, reference.locations, atol=1e-6)


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


def test_pack_passes_the_snr_amplitudes_through_to_float32(
    packed: Packed, reference: Reference
) -> None:
    reference_amplitude = reference.snrs[..., 0]
    np.testing.assert_allclose(
        packed.snrs[..., 0],
        reference_amplitude,
        rtol=0,
        atol=1e-6 * reference_amplitude.max(),
    )


def test_pack_phases_differ_from_bayestar_by_whole_turns_per_detector(
    packed: Packed, reference: Reference
) -> None:
    # ligo.skymap.bayestar.filter.unwrap flattens the whole array, so phase
    # continues from one detector into the next; injstar unwraps each row.
    turns = (reference.snrs[..., 1] - packed.snrs[..., 1]) / (2 * np.pi)
    np.testing.assert_allclose(turns, np.round(turns), atol=1e-5)
    np.testing.assert_allclose(
        turns, np.broadcast_to(turns[:, :1], turns.shape), atol=1e-5
    )
