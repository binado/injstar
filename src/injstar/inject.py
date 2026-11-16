r"""SNR-level simulation.

Batched JAX code that builds each detector's frequency-domain data, the
injected signal plus Gaussian noise, and matched-filters it against a template
with one inverse FFT per detector, plus detection flags.

The signal and the template are separate waveform models. The template is the
face-on plus polarization :math:`g(f)` of the template model at 1 Mpc, which
must be a dominant-mode aligned-spin model because BAYESTAR's signal model is
the dominant mode. With :math:`S_j` the one-sided PSD of detector :math:`j`:

.. math::

    \sigma_j^2 = 4 \, \Delta f \sum_f \frac{|g(f)|^2}{S_j(f)}, \qquad
    \mathrm{horizon}_j = \sigma_j \ [\mathrm{Mpc}], \qquad
    \hat g_j = g / \sigma_j

The signal is the full projected strain of the signal model, evaluated at the
event's distance, inclination and phase, delayed to the arrival time
:math:`t_j` at the detector:

.. math::

    s_j(f) = \left[ F_+ h_+(f) + F_\times h_\times(f) \right]
    e^{-2 \pi i f t_j}, \qquad
    \rho_j = \sqrt{4 \, \Delta f \sum_f \frac{|s_j(f)|^2}{S_j(f)}}

.. math::

    z_j(t) = \langle \hat g_j, s_j \rangle(t) + n_j(t), \qquad
    \langle n \, n^* \rangle = 2 \, \delta

where :math:`\langle \hat g_j, s_j \rangle(t)` is the noise-weighted
inner product of the template with the signal shifted by :math:`t`. When the
signal model is the template model, :math:`h_+ = \tfrac{1+\cos^2\iota}{2} g/d_L`
and :math:`h_\times = -i \cos\iota \, g/d_L`, so the series peaks at
:math:`\rho_j = (\mathrm{horizon}_j / d_L) \left[ F_+ \tfrac{1+\cos^2\iota}{2}
- i F_\times \cos\iota \right]`, which is BAYESTAR's signal model
(``bayestar_signal_amplitude_model`` in ligo.skymap). Otherwise
:math:`\rho_j` bounds the series by the Cauchy-Schwarz inequality.

The rectangle sum is used for :math:`\sigma_j` so that the series of the
unit template peaks at exactly 1 on the grid; ligo.skymap's own horizon uses a
trapezoid rule, which differs only by half of the two band-edge bins.

The matched-filter quantities are the pure functions of
:mod:`injstar.matched_filter`. Times are offsets from each event's geocentric
coalescence time. JAX must run in 64-bit mode: call
:func:`injstar.runtime.configure_runtime` first.
"""

from collections.abc import Callable, Iterator, Mapping

import jax
import jax.numpy as jnp
import lal
import numpy as np
from numpy.typing import NDArray

from injstar import matched_filter
from injstar.dataset import ARRAY_FIELDS, InjectionResult
from injstar.detector import projection
from injstar.detector.network import DetectorRegistry
from injstar.detector.noise import simulate_fd_noise
from injstar.waveform import WaveformGenerator

#: Per-event inputs that do not depend on the waveform models, each an array over
#: events.
EXTRINSIC_FIELDS = (
    "right_ascension",
    "declination",
    "polarization_angle",
    "gmst",
)

#: The template is the waveform seen face-on, at 1 Mpc and zero phase.
_FACE_ON = {"distance": 1.0, "inclination": 0.0, "coa_phase": 0.0}


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


def _prepare_kernel(
    *,
    waveform: WaveformGenerator[jax.Array],
    template: WaveformGenerator[jax.Array],
    network: DetectorRegistry,
    minimum_frequency: float = 20.0,
    maximum_frequency: float = 1024.0,
    sampling_frequency: float = 4096.0,
    segment_duration: float = 8.0,
    crop_duration: float = 0.1,
    noise: bool = True,
    seed: int = 0,
    batch_size: int = 32,
    detector_snr: float = 4.0,
    network_snr: float = 8.0,
    min_detectors: int = 2,
) -> tuple[
    tuple[str, ...],
    Callable[[Mapping[str, jax.Array], jax.Array], dict[str, jax.Array]],
]:
    """Prepare the network once and compile the vectorized single-event kernel."""
    if not jax.config.jax_enable_x64:
        raise RuntimeError(
            "JAX must run in 64-bit mode; call injstar.runtime.configure_runtime()"
        )
    if (
        getattr(template, "is_precessing", False)
        or getattr(template, "highest_mode", 2) > 2
    ):
        raise ValueError(
            "template is precessing or has higher modes; set template_approximant "
            "to a dominant-mode aligned-spin model such as IMRPhenomXAS"
        )
    names, responses, locations = network.arrays(tuple(network.detectors))
    bins = matched_filter.band_bins(
        minimum_frequency=minimum_frequency,
        maximum_frequency=maximum_frequency,
        sampling_frequency=sampling_frequency,
        segment_duration=segment_duration,
    )
    if len(bins) < 3:
        raise ValueError("the frequency band holds fewer than three bins")
    delta_f = 1.0 / segment_duration
    frequencies = jnp.asarray(bins * delta_f)
    psd = np.stack(
        [network.detectors[name].sensitivity(np.asarray(frequencies)) for name in names]
    )
    inverse = np.where(np.isfinite(psd), 1.0 / psd, 0.0)
    empty = [name for name, row in zip(names, inverse, strict=True) if not row.any()]
    if empty:
        raise ValueError(f"PSD does not cover the frequency band for {empty}")
    inverse_psd = jnp.asarray(inverse)
    n_samples = round(sampling_frequency * segment_duration)
    half = (2 * int(crop_duration * sampling_frequency / 2) + 1) // 2
    root = jax.random.key(seed)
    template_generator = template

    def _event_kernel(
        event: Mapping[str, jax.Array], row: jax.Array
    ) -> dict[str, jax.Array]:
        n, df, rate = n_samples, delta_f, sampling_frequency

        # The template: the face-on plus polarization at 1 Mpc.
        template, _ = template_generator(
            frequencies,
            {name: event[name] for name in template_generator.parameter_names}
            | _FACE_ON,
        )
        sigma = matched_filter.optimal_snr(
            template, inverse_psd=inverse_psd, delta_f=df
        )
        bandwidth = jnp.sqrt(
            matched_filter.second_frequency_cumulant(
                template, frequencies=frequencies, inverse_psd=inverse_psd
            )
        )
        unit_template = template / jnp.where(sigma > 0, sigma, 1.0)[:, None]

        sky = {
            "right_ascension": event["right_ascension"],
            "declination": event["declination"],
        }
        antenna_plus, antenna_cross = jax.vmap(
            lambda response: projection.antenna_pattern(
                response,
                event["gmst"],
                polarization_angle=event["polarization_angle"],
                **sky,
            )
        )(jnp.asarray(responses))
        delay = jax.vmap(
            lambda location: projection.time_delay_from_geocenter(
                location, event["gmst"], **sky
            )
        )(jnp.asarray(locations))

        # The signal: the full projected strain, with the geocentric time on
        # sample n // 2 of the periodic segment.
        plus, cross = waveform(
            frequencies,
            {name: event[name] for name in waveform.parameter_names},
        )
        peak_time = (n // 2) / rate + delay
        phase = -2 * jnp.pi * frequencies * peak_time[:, None]
        data = (antenna_plus[:, None] * plus + antenna_cross[:, None] * cross) * (
            jnp.cos(phase) + 1j * jnp.sin(phase)
        )
        expected = matched_filter.optimal_snr(data, inverse_psd=inverse_psd, delta_f=df)
        if noise:
            # The key depends only on the root key and the 64-bit row index.
            high = (row >> 32).astype(jnp.uint32)
            low = (row & 0xFFFFFFFF).astype(jnp.uint32)
            key = jax.random.fold_in(jax.random.fold_in(root, high), low)
            data = data + simulate_fd_noise(key, inverse_psd=inverse_psd, delta_f=df)
        series = matched_filter.matched_filter_snr_series(
            data,
            unit_template,
            inverse_psd=inverse_psd,
            minimum_frequency=minimum_frequency,
            maximum_frequency=maximum_frequency,
            sampling_frequency=sampling_frequency,
            segment_duration=segment_duration,
        )

        centre = n // 2 + jnp.round(delay * rate).astype(jnp.int64)
        offsets = jnp.arange(-half, half + 1)
        crop = jnp.take_along_axis(series, centre[:, None] + offsets, axis=-1)
        observed = jnp.max(jnp.abs(crop), axis=-1)
        network = jnp.sqrt(jnp.sum(observed**2))
        detected = (jnp.sum(observed >= detector_snr) >= min_detectors) & (
            network >= network_snr
        )

        return {
            "expected_snr": expected,
            "observed_snr": observed,
            "horizon": sigma,
            "bandwidth": bandwidth,
            "network_snr": network,
            "detected": detected,
            "series": crop,
            "epochs": (centre - half - n // 2) / rate,
        }

    return names, jax.jit(jax.vmap(_event_kernel))


def _inject_batches(
    events: Mapping[str, NDArray],
    *,
    waveform: WaveformGenerator[jax.Array],
    template: WaveformGenerator[jax.Array],
    network: DetectorRegistry,
    minimum_frequency: float = 20.0,
    maximum_frequency: float = 1024.0,
    sampling_frequency: float = 4096.0,
    segment_duration: float = 8.0,
    crop_duration: float = 0.1,
    noise: bool = True,
    seed: int = 0,
    batch_size: int = 32,
    detector_snr: float = 4.0,
    network_snr: float = 8.0,
    min_detectors: int = 2,
    rows: NDArray | None = None,
) -> Iterator[InjectionResult]:
    """Yield complete batches, reusing one prepared callable and trimming padding."""
    names, kernel = _prepare_kernel(
        waveform=waveform,
        template=template,
        network=network,
        minimum_frequency=minimum_frequency,
        maximum_frequency=maximum_frequency,
        sampling_frequency=sampling_frequency,
        segment_duration=segment_duration,
        crop_duration=crop_duration,
        noise=noise,
        seed=seed,
        batch_size=batch_size,
        detector_snr=detector_snr,
        network_snr=network_snr,
        min_detectors=min_detectors,
    )
    fields = tuple(
        dict.fromkeys(
            (*EXTRINSIC_FIELDS, *waveform.parameter_names, *template.parameter_names)
        )
    )
    missing = [name for name in fields if name not in events]
    if missing:
        raise KeyError(f"events lack the fields {missing}")
    if rows is None:
        rows = np.arange(len(next(iter(events.values()))), dtype=np.uint64)
    count, size = len(rows), batch_size
    for start in range(0, count, size):
        index = np.minimum(np.arange(start, start + size), count - 1)
        valid = min(size, count - start)
        batch = {
            name: jnp.asarray(events[name][index], dtype=jnp.float64) for name in fields
        }
        output = kernel(batch, jnp.asarray(rows[index], dtype=jnp.uint64))
        yield InjectionResult(
            **{name: values[:valid] for name, values in output.items()},
            row_index=jnp.asarray(rows[start : start + valid], dtype=jnp.uint64),
            detectors=names,
            sampling_frequency=sampling_frequency,
        )


def inject_events(
    events: Mapping[str, NDArray],
    *,
    waveform: WaveformGenerator[jax.Array],
    template: WaveformGenerator[jax.Array],
    network: DetectorRegistry,
    minimum_frequency: float = 20.0,
    maximum_frequency: float = 1024.0,
    sampling_frequency: float = 4096.0,
    segment_duration: float = 8.0,
    crop_duration: float = 0.1,
    noise: bool = True,
    seed: int = 0,
    batch_size: int = 32,
    detector_snr: float = 4.0,
    network_snr: float = 8.0,
    min_detectors: int = 2,
    rows: NDArray | None = None,
) -> InjectionResult:
    """Simulate injections, returning complete batched SNR and series results.

    Parameters
    ----------
    events
        Arrays with a leading injection axis. Masses are detector-frame in
        solar masses, distances in Mpc and angles in radians. Inputs include
        ``EXTRINSIC_FIELDS`` and both generators' parameter names.
    waveform, template
        Explicit signal and dominant-mode aligned-spin template generators.
    network
        Selected registry; its detectors define the detector axis in registry order.
    minimum_frequency, maximum_frequency, sampling_frequency
        Frequency band and sample rate in Hz.
    segment_duration, crop_duration
        Segment and crop durations in seconds.
    noise, seed, batch_size
        Noise generation and batching settings.
    detector_snr, network_snr, min_detectors
        Detection thresholds.
    rows
        Original population indices for noise seeding; defaults to input order.

    Returns
    -------
    InjectionResult
        Arrays with a leading injection axis, even for one injection.
    """
    batches = list(
        _inject_batches(
            events,
            waveform=waveform,
            template=template,
            network=network,
            minimum_frequency=minimum_frequency,
            maximum_frequency=maximum_frequency,
            sampling_frequency=sampling_frequency,
            segment_duration=segment_duration,
            crop_duration=crop_duration,
            noise=noise,
            seed=seed,
            batch_size=batch_size,
            detector_snr=detector_snr,
            network_snr=network_snr,
            min_detectors=min_detectors,
            rows=rows,
        )
    )
    return InjectionResult(
        **{
            name: jnp.concatenate([getattr(batch, name) for batch in batches])
            for name in ARRAY_FIELDS
        },
        detectors=batches[0].detectors,
        sampling_frequency=sampling_frequency,
    )
