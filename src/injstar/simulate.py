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
from injstar.config import RunConfig
from injstar.dataset import ARRAY_FIELDS, InjectionResult
from injstar.detector import projection
from injstar.detector.network import DetectorRegistry
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


def _prepare_kernel(
    *,
    waveform: WaveformGenerator[jax.Array],
    template: WaveformGenerator[jax.Array],
    network: DetectorRegistry,
    config: RunConfig,
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
    names, responses, locations = network.arrays(config.network)
    bins = matched_filter.band_bins(
        minimum_frequency=config.minimum_frequency,
        maximum_frequency=config.maximum_frequency,
        sampling_frequency=config.sampling_frequency,
        segment_duration=config.segment_duration,
    )
    if len(bins) < 3:
        raise ValueError("the frequency band holds fewer than three bins")
    delta_f = 1.0 / config.segment_duration
    frequencies = jnp.asarray(bins * delta_f)
    psd = np.stack(
        [network.detectors[name].sensitivity(np.asarray(frequencies)) for name in names]
    )
    inverse = np.where(np.isfinite(psd), 1.0 / psd, 0.0)
    empty = [name for name, row in zip(names, inverse, strict=True) if not row.any()]
    if empty:
        raise ValueError(f"PSD does not cover the frequency band for {empty}")
    inverse_psd = jnp.asarray(inverse)
    n_samples = round(config.sampling_frequency * config.segment_duration)
    half = config.crop_samples // 2
    root = jax.random.key(config.seed)
    template_generator = template

    def _event_kernel(
        event: Mapping[str, jax.Array], row: jax.Array
    ) -> dict[str, jax.Array]:
        n, df, rate = n_samples, delta_f, config.sampling_frequency

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
        if config.noise:
            # The key depends only on the root key and the 64-bit row index.
            high = (row >> 32).astype(jnp.uint32)
            low = (row & 0xFFFFFFFF).astype(jnp.uint32)
            key = jax.random.fold_in(jax.random.fold_in(root, high), low)
            draw = jax.random.normal(key, (*inverse_psd.shape, 2))
            data = data + _noise_spectrum(
                draw[..., 0] + 1j * draw[..., 1], inverse_psd=inverse_psd, delta_f=df
            )
        series = matched_filter.matched_filter_snr_series(
            data,
            unit_template,
            inverse_psd=inverse_psd,
            minimum_frequency=config.minimum_frequency,
            maximum_frequency=config.maximum_frequency,
            sampling_frequency=config.sampling_frequency,
            segment_duration=config.segment_duration,
        )

        centre = n // 2 + jnp.round(delay * rate).astype(jnp.int64)
        offsets = jnp.arange(-half, half + 1)
        crop = jnp.take_along_axis(series, centre[:, None] + offsets, axis=-1)
        observed = jnp.max(jnp.abs(crop), axis=-1)
        network = jnp.sqrt(jnp.sum(observed**2))
        detected = (
            jnp.sum(observed >= config.detection.detector_snr)
            >= config.detection.min_detectors
        ) & (network >= config.detection.network_snr)

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
    config: RunConfig,
    rows: NDArray | None = None,
) -> Iterator[InjectionResult]:
    """Yield complete batches, reusing one prepared callable and trimming padding."""
    names, kernel = _prepare_kernel(
        waveform=waveform, template=template, network=network, config=config
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
    count, size = len(rows), config.batch_size
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
            sampling_frequency=config.sampling_frequency,
        )


def inject_events(
    events: Mapping[str, NDArray],
    *,
    waveform: WaveformGenerator[jax.Array],
    template: WaveformGenerator[jax.Array],
    network: DetectorRegistry,
    config: RunConfig,
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
        Detector registry containing the configured network.
    config
        Band, sampling, detection, noise seed and batch size settings.
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
            config=config,
            rows=rows,
        )
    )
    return InjectionResult(
        **{
            name: jnp.concatenate([getattr(batch, name) for batch in batches])
            for name in ARRAY_FIELDS
        },
        detectors=batches[0].detectors,
        sampling_frequency=config.sampling_frequency,
    )
