# Step 0 audit

Checked on 2026-11-12 against the versions below. This is a source read plus one
toy call of the numerical core. No physics has been validated: nothing here says
that a simulated event localizes correctly.

## Pinned versions

Resolved in `uv.lock`; `pyproject.toml` carries them as lower bounds.

| Package | Version | Note |
| --- | --- | --- |
| Python | >= 3.12 | jax 0.12 and ripplegw 0.5.1 both require 3.12 |
| jax, jaxlib | 0.12.0.1 | |
| ripplegw | 0.5.1 | ripple's distribution name on PyPI |
| ligo.skymap | 2.5.4 | tag `v2.5.4`, commit `413d2b4c9daa01ade2099ee9cf8341dafa4a372f` |
| astropy | 8.0.2 | |
| lalsuite | 8.0.2 | pulled in by ligo.skymap |
| numpy | 2.3.5 | |
| scipy | 1.16.3 | held below 1.18 by the `validation` extra |
| pycbc | 2.11.0 | `validation` extra |
| gwmock-signal | 0.24.0 | `validation` extra, Python >= 3.13 only |

Packaging notes:

- ligo.skymap ships `cp311-abi3` wheels for macOS arm64, macOS x86_64 and Linux
  (x86_64, aarch64), so the compiled core installs on Python 3.12 to 3.14 without
  a build.
- The lockfile is universal, so the `validation` extra constrains the versions
  everyone gets. Adding it moved scipy from 1.18.1 to 1.16.3. Drop the extra from
  the lock if that becomes a problem.
- PyCBC has wheels up to CPython 3.13. On 3.14 the extra needs a source build.
- The lock written by uv 0.13.5 passes `uv lock --check` under uv 0.12.18, the
  version CI pins.

## Findings

Paths under `ligo/skymap/` are in the installed package; `src/` paths are in the
ligo.skymap repository at the tag above.

| Claim in the design | Result | Source |
| --- | --- | --- |
| `bayestar_realize_coincs` simulates at the SNR level | Confirmed | `tool/bayestar_realize_coincs.py`, `simulate_snr`, lines 178-242 |
| Distance integral is rescaled to the event SNR | Corrected | `src/bayestar_sky_map.c`, lines 270-520 and 915-931 |
| Sky search starts at nside 16 | Confirmed | `src/bayestar_sky_map.c`, line 944 (`order0 = 4`) |
| Maximum resolution is nside 2048 | Confirmed | `src/bayestar_sky_map.c`, lines 1006-1010 |
| C layer casts SNRs and responses to float32, the rest to float64 | Confirmed | `src/core.c`, lines 820-854 |
| Wrapper default `rescale_loglikelihood` is 0.83 | Confirmed | `bayestar/__init__.py`, line 62 |
| Cosmological prior is marked unreviewed | Confirmed | `bayestar/__init__.py`, `condition_prior` |
| Expected-SNR formula has `- i * Fx * cos i` and a phase factor | Corrected | `tool/bayestar_realize_coincs.py`, line 190 |
| No dependency enables JAX x64 on import | Confirmed | see "JAX precision" |

### SNR-level simulation in `bayestar_realize_coincs`

`simulate_snr` never builds strain. Per detector it computes:

```python
z = (0.5 * (1 + u2) * Fplus + 1j * u * Fcross) * horizon / distance
snr_series = z * np.concatenate((acor[:0:-1].conj(), acor))
```

- `u = cos(inclination)`, `Fplus, Fcross = lal.ComputeDetAMResponse(response, ra,
  dec, psi, gmst)`, `horizon = SignalModel(HS).get_horizon_distance()`.
- The cross term enters with `+1j`. The design had `-i`; it is corrected.
- There is no coalescence-phase factor. `coa_phase` is read and commented out as
  arbitrary.
- With `measurement_error="gaussian-noise"`, coloured noise is drawn in the
  frequency domain with amplitude proportional to `sqrt(HS / sigmasq)` and
  transformed by the same `filter.autocorrelation` routine with
  `normalize=False`. The default is `"zero-noise"`.
- The arrival time is applied by cubic interpolation to the nearest sample, and
  the series is then re-centred on the sample of maximum amplitude. The design's
  exact phase ramp is not what the reference does, so comparisons in step 1 will
  agree only to sub-sample accuracy unless the injection time is placed on a
  sample.
- The default SNR-series duration is 0.1 s.

### Horizon and autocorrelation

- `SignalModel.get_horizon_distance()` returns `sqrt(4 * integral(|h|^2 / S df))`
  using a trapezoid rule over the nonzero band (`bayestar/filter.py`, lines
  552-573). The design writes a rectangle sum; the two differ at the band edges.
- `filter.autocorrelation` zero-pads the one-sided power series to twice its
  length, so the SNR-series sample rate is `2 * len(H) * deltaF`. It is set by the
  template's frequency grid, not chosen freely (`bayestar/filter.py`, lines
  356-398).
- Unresolved: the reference distance of `filter.sngl_inspiral_psd`. The horizon
  formula implies 1 Mpc, but the function was not read.

### Conditioning and centring (`bayestar.condition`)

- Detector locations are divided by `c` and centred on a weighted mean, with
  weights `1 / sigma_t^2` from each detector's Cramér-Rao timing uncertainty at
  its measured SNR. It is not the plain centroid.
- The reference epoch is the same weighted mean of the arrival times, kept in
  integer nanoseconds. `gmst` is `lal.GreenwichMeanSiderealTime(epoch)`.
- `epochs` passed to the core are the first-sample times relative to that epoch:
  `toa - 0.5 * (nsamples - 1) / sample_rate`.
- Crop half-width: the larger of (greatest distance from the weighted centre to a
  detector, in light-seconds, plus 5 ms) and a quarter period of the widest
  autocorrelation, `0.5 * pi * sigma_t(SNR=1)`. Length is
  `2 * ceil(max_abs_t * sample_rate) - 1`, always odd.
- All series must share one sample rate and one odd length.
- `snrs` is `stack((abs(z), unwrap(angle(z))), axis=-1)`.
- If `max_distance` is not given, the wrapper uses `max(horizons) / 4`. For ET and
  CE that default is far too small or too large depending on the source; the run
  config must set the bounds explicitly.

### Distance integral

The design said the integral is rescaled to the event's SNR. It is not.

- Three lookup tables (one per distance moment) are built on every call, each
  400 x 400, bicubic in `(log p, log r0)`.
- The table's upper amplitude bound is `pmax = rescale * sqrt(0.5 * sum(horizons^2))`.
  It depends on the network, not on the event.
- When each table entry is computed, the integrand is split at the likelihood
  peak `r = 2 p^2 / b` and at the points where it falls to 1% of the peak, and
  normalised to its maximum. That adapts the quadrature to a narrow peak.
- At evaluation the term `(b / 2p)^2`, the log-likelihood at the peak, is added
  analytically, so the table interpolates a smooth remainder.

Accuracy of the table at network SNR in the hundreds is not established by
reading the code. It belongs in the high-SNR gate.

### Sky sampling

- Order 4 (nside 16, 3072 pixels) is evaluated in full.
- For each of 7 levels the 768 highest-probability pixels are split in four,
  adding 2304 pixels per level, up to order 11 (nside 2048).
- Every map therefore has `3072 + 7 * 2304 = 19200` pixels whatever the SNR. The
  toy call returned exactly 19200 rows.
- At most 3072 pixels are at order 11, covering about 2.5 deg^2. A pixel there is
  about 8.2e-4 deg^2.
- Marginalisation uses 10 polarization and 10 inclination nodes, fixed.

Consequence for storage: with five float64/int64 columns an uncompressed
multi-order map is under 1 MB per event. This is an estimate from the row count,
not a measured file size.

### `core.toa_phoa_snr` interface

The docstring is the placeholder "fill me in"; the signature comes from
`src/core.c`, lines 752-930.

```text
toa_phoa_snr(min_distance, max_distance, prior_distance_power, cosmology, gmst,
             sample_rate, epochs, snrs, responses, locations, horizons,
             rescale_loglikelihood) -> (pixels, log_bci, log_bsn)
```

- Format string `ddiidfOOOOOf`: `sample_rate` and `rescale_loglikelihood` are C
  floats, `cosmology` is an int. All twelve arguments are required.
- `epochs`, `locations`, `horizons` are read as float64; `snrs` and `responses`
  as float32.
- `pixels` is a structured array with fields `UNIQ`, `PROBDENSITY`, `DISTMEAN`,
  `DISTSTD`. `DISTMU`, `DISTSIGMA` and `DISTNORM` are produced afterwards by
  `distance.moments_to_parameters` in `bayestar.localize`.
- `cosmology=True` is accepted only with `prior_distance_power=2`.

Toy call: three detectors with made-up diagonal response tensors, 65 samples at
4096 Hz, a Gaussian amplitude peak of 10, zero phase. It returned in 1.7 s on
four threads with probability summing to 1. The input is not physical; this shows
only that the call runs and what it returns.

### JAX precision

- `jax.config.jax_enable_x64` is `False` after `import jax` and still `False`
  after `import ripplegw`.
- The only `jax_enable_x64` update in ripplegw is in
  `ripplegw/benchmarks/timings/timing.py`, which is not imported by the package.

injstar must enable x64 itself, as the design says.

## Not covered

- gwmock-signal and PyCBC were installed but not inspected.
- ripplegw's waveform amplitude interface and its reference distance.
- `bayestar.localize` postprocessing was read only as far as the distance
  parameters and marginal moments.
- Nothing was run on Linux or on Python 3.12 and 3.14; CI will be the first test.
