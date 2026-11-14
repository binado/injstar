r"""SNR-level simulation.

Batched JAX code that builds each detector's frequency-domain data, the
template at its expected SNR plus Gaussian noise, and matched-filters it against
the template with one inverse FFT per detector, plus detection flags.

For detector :math:`j`, with :math:`|h(f)|` the face-on amplitude at 1 Mpc and
:math:`S_j` the one-sided PSD:

.. math::

    \sigma_j^2 = 4 \, \Delta f \sum_f \frac{|h(f)|^2}{S_j(f)}, \qquad
    \mathrm{horizon}_j = \sigma_j \ [\mathrm{Mpc}]

.. math::

    \rho_j = \frac{\mathrm{horizon}_j}{d_L}
    \left[ F_+ \frac{1 + \cos^2 \iota}{2} - i \, F_\times \cos \iota \right]

.. math::

    z_j(t) = \rho_j \, A_j(t - t_j) + n_j(t), \qquad
    \langle n \, n^* \rangle = 2 A_j

The sign of the cross term is BAYESTAR's signal model
(``bayestar_signal_amplitude_model`` in ligo.skymap), which ripple's
:math:`h_\times = -i \cos\iota \, h_+` also gives.

The rectangle sum is used for :math:`\sigma_j` so that :math:`A_j(0) = 1` exactly
on the grid; ligo.skymap's own horizon uses a trapezoid rule, which differs only by
half of the two band-edge bins.

The matched-filter quantities are the pure functions of
:mod:`injstar.matched_filter`. Times are offsets from each event's geocentric
coalescence time. JAX must run in 64-bit mode: call
:func:`injstar.runtime.configure_runtime` first.
"""

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Self

import jax
import jax.numpy as jnp
import lal
import numpy as np
from numpy.typing import NDArray

from injstar import matched_filter
from injstar.config import RunConfig
from injstar.detector import projection
from injstar.detector.network import DetectorRegistry
from injstar.waveform import Waveform
from injstar.waveform import waveform as make_waveform

#: Per-event inputs that do not depend on the waveform model, each an array over
#: events. ``inclination`` and ``distance`` are listed because the projection onto
#: the detectors is still written by hand rather than taken from the model.
EXTRINSIC_FIELDS = (
    "event_id",
    "right_ascension",
    "declination",
    "polarization_angle",
    "gmst",
    "inclination",
    "distance",
)

#: Per-event outputs shared by :meth:`Simulator.snr` and :meth:`Simulator.series`,
#: matching the fields of :class:`SnrResult`.
SCALAR_FIELDS = (
    "expected_snr",
    "observed_snr",
    "horizon",
    "bandwidth",
    "network_snr",
    "detected",
)


@dataclass(frozen=True)
class SnrResult:
    """Scalar outputs for a set of events; arrays lead with the event axis.

    Attributes
    ----------
    expected_snr
        ``|rho_j|``, shape ``(events, detectors)``.
    observed_snr
        Peak ``|z_j|`` of the cropped series, shape ``(events, detectors)``.
    horizon
        SNR = 1 distances in Mpc, shape ``(events, detectors)``.
    bandwidth
        RMS angular-frequency bandwidth in rad/s, shape
        ``(events, detectors)``; the timing uncertainty at SNR ``s`` is
        ``1 / (s * bandwidth)``.
    network_snr
        Root sum square of ``observed_snr``, shape ``(events,)``.
    detected
        Detection flag, shape ``(events,)``.
    """

    expected_snr: NDArray[np.float64]
    observed_snr: NDArray[np.float64]
    horizon: NDArray[np.float64]
    bandwidth: NDArray[np.float64]
    network_snr: NDArray[np.float64]
    detected: NDArray[np.bool_]


@dataclass(frozen=True)
class SeriesResult:
    """SNR time series for a set of events.

    Attributes
    ----------
    scalars
        The matching :class:`SnrResult`.
    series
        Complex SNR, shape ``(events, detectors, crop_samples)``.
    epochs
        Time of each series' first sample minus the event's geocentric
        coalescence time, in seconds, shape ``(events, detectors)``.
    """

    scalars: SnrResult
    series: NDArray[np.complex128]
    epochs: NDArray[np.float64]


def greenwich_sidereal_time(gps_times: NDArray[np.float64]) -> NDArray[np.float64]:
    """Greenwich mean sidereal time in radians at each GPS time.

    LAL's routine takes one :class:`lal.LIGOTimeGPS` at a time, hence the loop.
    It is used rather than the vectorised :mod:`astropy.time` sidereal time
    because BAYESTAR computes GMST with LAL, and the two differ by ~1e-5 rad.
    The result is not reduced modulo :math:`2\\pi`, which is harmless as it only
    enters trigonometric functions.

    Parameters
    ----------
    gps_times
        GPS times in seconds.
    """
    return np.array(
        [lal.GreenwichMeanSiderealTime(lal.LIGOTimeGPS(float(t))) for t in gps_times],
        dtype=float,
    )


def _signal_spectrum(
    unit_template: jax.Array,
    *,
    rho: jax.Array,
    peak_time: jax.Array,
    frequencies: jax.Array,
) -> jax.Array:
    """Frequency-domain data holding the template at the given SNR and time.

    Matched-filtering it against the unit-norm template gives ``rho`` at
    ``peak_time``. Shape ``(detectors, bins)``.
    """
    phase = -2 * jnp.pi * frequencies * peak_time[:, None]
    return rho[:, None] * unit_template * (jnp.cos(phase) + 1j * jnp.sin(phase))


def _noise_spectrum(
    white: jax.Array, *, inverse_psd: jax.Array, delta_f: float
) -> jax.Array:
    """Frequency-domain Gaussian noise with one-sided PSD ``1 / inverse_psd``.

    ``white`` has independent standard normal real and imaginary parts. Bins
    without a PSD (``inverse_psd == 0``) carry no noise.
    """
    psd = jnp.where(
        inverse_psd > 0, 1.0 / jnp.where(inverse_psd > 0, inverse_psd, 1.0), 0.0
    )
    return jnp.sqrt(psd / (4 * delta_f)) * white


class Simulator:
    """SNR-level simulator for one network and one band.

    Build it from a run configuration with :meth:`from_config`.

    Parameters
    ----------
    names
        Detector names.
    responses
        Earth-fixed response tensors, shape ``(detectors, 3, 3)``.
    locations
        Earth-fixed positions in metres, shape ``(detectors, 3)``.
    psd
        One-sided PSD of each detector on the bins of
        :func:`injstar.matched_filter.band_bins`, shape ``(detectors, bins)``;
        non-finite where it is undefined.
    minimum_frequency, maximum_frequency
        Frequency band of the matched filter in Hz.
    sampling_frequency
        Sample rate of the SNR series in Hz.
    segment_duration
        Length in seconds of the periodic segment; the frequency spacing is its
        inverse.
    crop_samples
        Odd number of samples in the cropped SNR series.
    waveform
        Waveform model; its face-on plus polarization is the template.
    detector_snr, network_snr, min_detectors
        Detection rule: at least ``min_detectors`` detectors reach
        ``detector_snr`` and the network SNR reaches ``network_snr``.
    noise
        Whether to add Gaussian noise to the SNR series.
    seed
        Seed of the noise.
    batch_size
        Events simulated per jitted call.
    """

    def __init__(
        self,
        *,
        names: Sequence[str],
        responses: NDArray[np.float64],
        locations: NDArray[np.float64],
        psd: NDArray[np.float64],
        minimum_frequency: float,
        maximum_frequency: float,
        sampling_frequency: float,
        segment_duration: float,
        crop_samples: int,
        waveform: Waveform,
        detector_snr: float,
        network_snr: float,
        min_detectors: int,
        noise: bool,
        seed: int,
        batch_size: int,
    ) -> None:
        if not jax.config.jax_enable_x64:
            raise RuntimeError(
                "JAX must run in 64-bit mode; call injstar.runtime.configure_runtime()"
            )
        self.names = tuple(names)
        self.minimum_frequency = minimum_frequency
        self.maximum_frequency = maximum_frequency
        self.sampling_frequency = sampling_frequency
        self.segment_duration = segment_duration
        self.n_samples = round(sampling_frequency * segment_duration)
        self.delta_f = 1.0 / segment_duration
        self.crop_samples = crop_samples
        self.noise = noise
        self.batch_size = batch_size
        bins = matched_filter.band_bins(
            minimum_frequency=minimum_frequency,
            maximum_frequency=maximum_frequency,
            sampling_frequency=sampling_frequency,
            segment_duration=segment_duration,
        )
        self.frequencies = bins * self.delta_f
        if psd.shape[-1] != len(bins):
            raise ValueError(
                f"psd has {psd.shape[-1]} bins but the band holds {len(bins)}"
            )

        self._inverse_psd = np.where(np.isfinite(psd), 1.0 / psd, 0.0)
        empty = [
            name
            for name, row in zip(self.names, self._inverse_psd, strict=True)
            if not row.any()
        ]
        if empty:
            raise ValueError(f"PSD does not cover the frequency band for {empty}")

        self._responses = responses
        self._locations = locations
        self._half = crop_samples // 2
        self._waveform = waveform
        self._detector_snr = detector_snr
        self._network_snr = network_snr
        self._min_detectors = min_detectors
        self._root = jax.random.key(seed)
        self._batched = jax.jit(jax.vmap(self._event))

    @classmethod
    def from_config(cls, config: RunConfig, registry: DetectorRegistry) -> Self:
        """Build the simulator for a run configuration.

        Parameters
        ----------
        config
            Run configuration: band, sampling, detection rule, seed, batch size.
        registry
            Detector registry holding the configured network.

        Raises
        ------
        ValueError
            If the band holds fewer than three bins, the waveform model is
            precessing or has modes above the dominant one, or a detector's PSD
            does not cover the band.
        """
        names, responses, locations = registry.arrays(config.network)

        bins = matched_filter.band_bins(
            minimum_frequency=config.minimum_frequency,
            maximum_frequency=config.maximum_frequency,
            sampling_frequency=config.sampling_frequency,
            segment_duration=config.segment_duration,
        )
        if len(bins) < 3:
            raise ValueError("the frequency band holds fewer than three bins")
        frequencies = bins / config.segment_duration
        psd = np.stack(
            [registry.detectors[name].sensitivity(frequencies) for name in names]
        )

        waveform = make_waveform(
            config.waveform_approximant,
            reference_frequency=config.reference_frequency,
        )
        if waveform.is_precessing or waveform.highest_mode > 2:
            raise ValueError(
                f"waveform {config.waveform_approximant!r} is precessing or has "
                "higher modes; only dominant-mode aligned-spin models are supported"
            )
        return cls(
            names=names,
            responses=responses,
            locations=locations,
            psd=psd,
            minimum_frequency=config.minimum_frequency,
            maximum_frequency=config.maximum_frequency,
            sampling_frequency=config.sampling_frequency,
            segment_duration=config.segment_duration,
            crop_samples=config.crop_samples,
            waveform=waveform,
            detector_snr=config.detection.detector_snr,
            network_snr=config.detection.network_snr,
            min_detectors=config.detection.min_detectors,
            noise=config.noise,
            seed=config.seed,
            batch_size=config.batch_size,
        )

    @property
    def event_fields(self) -> tuple[str, ...]:
        """Per-event inputs: the extrinsic ones and those the waveform reads."""
        return tuple(
            dict.fromkeys((*EXTRINSIC_FIELDS, *self._waveform.parameter_names))
        )

    def _event(self, event: Mapping[str, jax.Array]) -> dict[str, jax.Array]:
        """Simulate one event; every output leads with the detector axis."""
        frequencies = jnp.asarray(self.frequencies)
        inverse_psd = jnp.asarray(self._inverse_psd)
        n, df, rate = self.n_samples, self.delta_f, self.sampling_frequency

        # The template is the face-on plus polarization at 1 Mpc.
        source = {name: event[name] for name in self._waveform.parameter_names}
        template, _ = self._waveform.polarizations(
            frequencies,
            source | {"distance": 1.0, "inclination": 0.0, "coa_phase": 0.0},
        )
        sigma = matched_filter.optimal_snr(
            template, inverse_psd=inverse_psd, delta_f=df
        )
        bandwidth = jnp.sqrt(
            matched_filter.second_frequency_cumulant(
                template, frequencies=frequencies, inverse_psd=inverse_psd
            )
        )

        sky = {
            "right_ascension": event["right_ascension"],
            "declination": event["declination"],
        }
        plus, cross = jax.vmap(
            lambda response: projection.antenna_pattern(
                response,
                event["gmst"],
                polarization_angle=event["polarization_angle"],
                **sky,
            )
        )(jnp.asarray(self._responses))
        delay = jax.vmap(
            lambda location: projection.time_delay_from_geocenter(
                location, event["gmst"], **sky
            )
        )(jnp.asarray(self._locations))

        cos_inclination = jnp.cos(event["inclination"])
        rho = (sigma / event["distance"]) * (
            0.5 * (1 + cos_inclination**2) * plus - 1j * cos_inclination * cross
        )

        # The geocentric time sits on sample n // 2 of the periodic segment.
        peak_time = (n // 2) / rate + delay
        unit_template = template / jnp.where(sigma > 0, sigma, 1.0)[:, None]
        data = _signal_spectrum(
            unit_template, rho=rho, peak_time=peak_time, frequencies=frequencies
        )
        if self.noise:
            # The key depends only on the root key and the 64-bit event id.
            event_id = event["event_id"]
            high = (event_id >> 32).astype(jnp.uint32)
            low = (event_id & 0xFFFFFFFF).astype(jnp.uint32)
            key = jax.random.fold_in(jax.random.fold_in(self._root, high), low)
            draw = jax.random.normal(key, (*inverse_psd.shape, 2))
            data = data + _noise_spectrum(
                draw[..., 0] + 1j * draw[..., 1], inverse_psd=inverse_psd, delta_f=df
            )
        series = matched_filter.matched_filter_snr_series(
            data,
            unit_template,
            inverse_psd=inverse_psd,
            minimum_frequency=self.minimum_frequency,
            maximum_frequency=self.maximum_frequency,
            sampling_frequency=self.sampling_frequency,
            segment_duration=self.segment_duration,
        )

        centre = n // 2 + jnp.round(delay * rate).astype(jnp.int64)
        offsets = jnp.arange(-self._half, self._half + 1)
        crop = jnp.take_along_axis(series, centre[:, None] + offsets, axis=-1)
        observed = jnp.max(jnp.abs(crop), axis=-1)
        network = jnp.sqrt(jnp.sum(observed**2))
        detected = (jnp.sum(observed >= self._detector_snr) >= self._min_detectors) & (
            network >= self._network_snr
        )

        return {
            "expected_snr": jnp.abs(rho),
            "observed_snr": observed,
            "horizon": sigma,
            "bandwidth": bandwidth,
            "network_snr": network,
            "detected": detected,
            "series": crop,
            "epochs": (centre - self._half - n // 2) / rate,
        }

    def _batches(
        self, events: Mapping[str, NDArray]
    ) -> Iterator[tuple[int, dict[str, jax.Array]]]:
        """Fixed-size batches, the last padded by repeating its final event."""
        fields = self.event_fields
        missing = [name for name in fields if name not in events]
        if missing:
            raise KeyError(f"events lack the fields {missing}")
        count = len(events["event_id"])
        size = self.batch_size
        for start in range(0, count, size):
            index = np.minimum(np.arange(start, start + size), count - 1)
            batch = {
                name: jnp.asarray(
                    np.asarray(events[name])[index],
                    dtype=jnp.uint64 if name == "event_id" else jnp.float64,
                )
                for name in fields
            }
            yield min(size, count - start), batch

    def _run(self, events: Mapping[str, NDArray], keep: Sequence[str]) -> dict:
        chunks: dict[str, list[np.ndarray]] = {name: [] for name in keep}
        for valid, batch in self._batches(events):
            output = self._batched(batch)
            for name in keep:
                chunks[name].append(np.asarray(output[name])[:valid])
        return {name: np.concatenate(parts) for name, parts in chunks.items()}

    def snr(self, events: Mapping[str, NDArray]) -> SnrResult:
        """Expected and observed SNRs and detection flags.

        Parameters
        ----------
        events
            Arrays over events for each name in :attr:`event_fields`. Masses
            are detector-frame in solar masses, distance in Mpc, angles in
            radians; ``event_id`` is a non-negative integer. Other entries are
            ignored.
        """
        return SnrResult(**self._run(events, SCALAR_FIELDS))

    def series(self, events: Mapping[str, NDArray]) -> SeriesResult:
        """Cropped SNR time series, with the scalars of :meth:`snr`.

        Parameters
        ----------
        events
            As for :meth:`snr`.
        """
        output = self._run(events, (*SCALAR_FIELDS, "series", "epochs"))
        series, epochs = output.pop("series"), output.pop("epochs")
        return SeriesResult(SnrResult(**output), series, epochs)
