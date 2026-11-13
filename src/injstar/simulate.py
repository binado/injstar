r"""SNR-level simulation.

Batched JAX code that draws each detector's matched-filter SNR time series
directly: expected complex SNR, whitened template autocorrelation and correlated
Gaussian noise from one inverse FFT per detector, plus detection flags.

For detector :math:`j`, with :math:`|h(f)|` the face-on amplitude at 1 Mpc and
:math:`S_j` the one-sided PSD:

.. math::

    \sigma_j^2 = 4 \, \Delta f \sum_f \frac{|h(f)|^2}{S_j(f)}, \qquad
    \mathrm{horizon}_j = \sigma_j \ [\mathrm{Mpc}]

.. math::

    \rho_j = \frac{\mathrm{horizon}_j}{d_L}
    \left[ F_+ \frac{1 + \cos^2 \iota}{2} + i \, F_\times \cos \iota \right]

.. math::

    z_j(t) = \rho_j \, A_j(t - t_j) + n_j(t), \qquad
    \langle n \, n^* \rangle = 2 A_j

The rectangle sum is used for :math:`\sigma_j` so that :math:`A_j(0) = 1` exactly
on the grid; ligo.skymap's own horizon uses a trapezoid rule, which differs only by
half of the two band-edge bins.

Times are offsets from each event's geocentric coalescence time. JAX must run
in 64-bit mode: call :func:`injstar.runtime.configure_runtime` first.

Design: "SNR-level simulation" in docs/design.md.
"""

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import lal
import numpy as np
import ripplegw
from numpy.typing import NDArray

from injstar.config import RunConfig
from injstar.detector import projection
from injstar.detector.network import DetectorRegistry

#: Per-event inputs of the simulation, each an array over events.
EVENT_FIELDS = (
    "event_id",
    "chirp_mass",
    "symmetric_mass_ratio",
    "spin_1z",
    "spin_2z",
    "luminosity_distance",
    "right_ascension",
    "declination",
    "polarization_angle",
    "inclination",
    "gmst",
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


def _event_key(root: jax.Array, event_id: jax.Array) -> jax.Array:
    """A key that depends only on the root key and the 64-bit event id."""
    high = (event_id >> 32).astype(jnp.uint32)
    low = (event_id & 0xFFFFFFFF).astype(jnp.uint32)
    return jax.random.fold_in(jax.random.fold_in(root, high), low)


class Simulator:
    """SNR-level simulator for one network and one run configuration.

    Parameters
    ----------
    config
        Run configuration: band, sampling, detection rule, seed, batch size.
    registry
        Detector registry holding the configured network.
    """

    def __init__(self, config: RunConfig, registry: DetectorRegistry) -> None:
        if not jax.config.jax_enable_x64:
            raise RuntimeError(
                "JAX must run in 64-bit mode; call injstar.runtime.configure_runtime()"
            )
        self.config = config
        self.names, responses, locations = registry.arrays(config.network)

        self.n_samples = config.window_samples
        self.delta_f = 1.0 / config.window_duration
        first = int(np.ceil(config.f_low / self.delta_f))
        last = min(int(np.floor(config.f_high / self.delta_f)), self.n_samples // 2 - 1)
        if last - first < 2:
            raise ValueError("the frequency band holds fewer than three bins")
        self._bins = np.arange(first, last + 1)
        self.frequencies = self._bins * self.delta_f

        psd = np.stack(
            [
                registry.detectors[name].sensitivity(self.frequencies)
                for name in self.names
            ]
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
        self._half = config.crop_samples // 2
        waveform = ripplegw.waveform(config.waveform.model, f_ref=config.waveform.f_ref)
        if not isinstance(waveform, ripplegw.AmplitudePhaseWaveform):
            raise ValueError(
                f"waveform {config.waveform.model!r} does not expose an amplitude"
            )
        self._waveform = waveform
        self._root = jax.random.key(config.seed)
        self._batched = jax.jit(jax.vmap(self._event))

    def _event(self, event: Mapping[str, jax.Array]) -> dict[str, jax.Array]:
        """Simulate one event; every output leads with the detector axis."""
        config = self.config
        frequencies = jnp.asarray(self.frequencies)
        inverse_psd = jnp.asarray(self._inverse_psd)
        n, df, rate = self.n_samples, self.delta_f, config.sample_rate

        amplitude = self._waveform.amplitude(
            frequencies,
            {
                "M_c": event["chirp_mass"],
                "eta": event["symmetric_mass_ratio"],
                "s1_z": event["spin_1z"],
                "s2_z": event["spin_2z"],
                "d_L": 1.0,
                "phase_c": 0.0,
                "iota": 0.0,
            },
        )
        amplitude = jnp.nan_to_num(amplitude, nan=0.0, posinf=0.0, neginf=0.0)
        power = amplitude**2 * inverse_psd
        total = jnp.sum(power, axis=-1)
        safe_total = jnp.where(total > 0, total, 1.0)
        sigma = jnp.sqrt(4 * df * total)
        safe_sigma = jnp.where(sigma > 0, sigma, 1.0)

        omega = 2 * jnp.pi * frequencies
        mean = jnp.sum(omega * power, axis=-1) / safe_total
        mean_square = jnp.sum(omega**2 * power, axis=-1) / safe_total
        bandwidth = jnp.sqrt(jnp.maximum(mean_square - mean**2, 0.0))

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
        rho = (sigma / event["luminosity_distance"]) * (
            0.5 * (1 + cos_inclination**2) * plus + 1j * cos_inclination * cross
        )

        # The geocentric time sits on sample n // 2 of the periodic window.
        peak_time = (n // 2) / rate + delay
        phase = -2 * jnp.pi * frequencies * peak_time[:, None]
        spectrum = (
            (power / safe_total[:, None])
            * rho[:, None]
            * (jnp.cos(phase) + 1j * jnp.sin(phase))
        )
        if config.noise:
            draw = jax.random.normal(
                _event_key(self._root, event["event_id"]), (*power.shape, 2)
            )
            white = draw[..., 0] + 1j * draw[..., 1]
            spectrum = spectrum + (
                (2 * jnp.sqrt(df) / safe_sigma)[:, None]
                * amplitude
                * jnp.sqrt(inverse_psd)
                * white
            )

        full = jnp.zeros((len(self.names), n), dtype=spectrum.dtype)
        full = full.at[:, jnp.asarray(self._bins)].set(spectrum)
        series = n * jnp.fft.ifft(full, axis=-1)

        centre = n // 2 + jnp.round(delay * rate).astype(jnp.int64)
        offsets = jnp.arange(-self._half, self._half + 1)
        crop = jnp.take_along_axis(series, centre[:, None] + offsets, axis=-1)
        observed = jnp.max(jnp.abs(crop), axis=-1)
        network = jnp.sqrt(jnp.sum(observed**2))
        detection = config.detection
        detected = (
            jnp.sum(observed >= detection.detector_snr) >= detection.min_detectors
        ) & (network >= detection.network_snr)

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
        missing = [name for name in EVENT_FIELDS if name not in events]
        if missing:
            raise KeyError(f"events lack the fields {missing}")
        count = len(events["event_id"])
        size = self.config.batch_size
        for start in range(0, count, size):
            index = np.minimum(np.arange(start, start + size), count - 1)
            batch = {
                name: jnp.asarray(
                    np.asarray(events[name])[index],
                    dtype=jnp.uint64 if name == "event_id" else jnp.float64,
                )
                for name in EVENT_FIELDS
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
            Arrays over events for each name in :data:`EVENT_FIELDS`. Masses
            are detector-frame in solar masses, distance in Mpc, angles in
            radians; ``event_id`` is a non-negative integer.
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
