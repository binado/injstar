# Simulated sky–distance posteriors with BAYESTAR

## Objective

Build a standalone package that turns a population of simulated compact binaries
and a detector-network specification into one BAYESTAR sky–distance posterior per
detected event, for arbitrary networks including ET and CE. corrtracers is one
consumer among several: it reads the output files and has no dependency on the
package.

This is a design plan based on documentation and source inspection. No
interoperability test or benchmark has been run. The statements about the
ligo.skymap source were checked against ligo.skymap 2.5.4 in step 0; the findings
and source locations are in [audit.md](audit.md).

## Scope

In scope, and assumed throughout:

- Simulated injections only, filtered with the injected intrinsic parameters
  (known template).
- Stationary Gaussian noise, independent between detectors, with the filter PSD
  equal to the true PSD.
- Static detector response and a nonprecessing, dominant-mode waveform, matching
  BAYESTAR's likelihood.

Out of scope: real data, template-bank searches and trigger selection, template
mismatch. Earth rotation, higher modes, precession and correlated noise are
model-mismatch extensions (see "Strain-level path").

## Why simulate at the SNR level

Under the assumptions above, the matched-filter output of `d = s + n` is known in
closed form, noise included, because the filter is linear:

```text
z_j(t) = rho_j * A_j(t - t_j) + n_j(t)
```

- `rho_j` is the expected complex SNR in detector `j`.
- `A_j` is the whitened template autocorrelation, `A_j(0) = 1`.
- `n_j` is zero-mean complex Gaussian noise with `<n(t) n*(t')> = 2 A_j(t - t')`.

Drawing `z_j` directly is statistically identical to generating strain, adding
noise and matched filtering. It needs only `|h(f)|^2 / S_j(f)`, so there is no
waveform phase, no long strain buffer, no padding and no duration binning. Every
event has the same array shapes.

BAYESTAR's own injection tool `bayestar_realize_coincs` works this way (confirmed
in the audit) and is the validation reference. It places the signal with cubic
interpolation to the nearest sample, so agreement with the exact phase ramp used
here is approximate at the sub-sample level.

## Architecture

```text
Network spec + PSD files      Population file      Run config
          \                         |                  /
           v                        v                 v
        Host setup: response tensors, locations, PSD samples, grids
                                    |
                                    v
        JAX, batched: |h(f)|, sigma_j, horizons, expected SNR, arrival
        times, SNR series with correlated noise, detection flags
                                    |
                              jax.device_get
                                    |
                                    v
        ligo.skymap.core.toa_phoa_snr  (CPU, one call per detected event)
                                    |
                                    v
        Per-event multi-order FITS + summary table + run manifest
```

XML is not needed. `core.toa_phoa_snr` takes response tensors and locations
directly, so custom detectors need no LAL detector identifiers.

BAYESTAR is expected to dominate runtime. JAX is used because ripple provides the
waveform amplitude and batching is convenient, not because preprocessing is the
bottleneck.

## Inputs

All three are files, so the package is reusable across projects.

**Network spec** (TOML, the detector registry schema of astrogwb), per detector:

- Name, vertex latitude, longitude and height, arm azimuths and tilts, in
  radians and metres. Arms need not be orthogonal (ET uses 60 degrees).
- PSD reference (packaged curve, path or URL), with an explicit `asd` or `psd`
  kind.
- Geometry and PSD are independent entries: a CE noise curve does not define a CE
  site.

**Population file** (CSV or HDF5 in gwmock-pop's layout), per event:

- Event id, detector-frame masses, aligned spins, luminosity distance in Mpc,
  RA, Dec, inclination, polarization, reference phase, geocentric GPS time.
- If source-frame masses and redshift are given instead, the cosmology used to
  convert them is part of the run config.

**Run config:**

- Waveform model, frequency band, sample rate, SNR-series window and crop length.
- Detection rule (per-detector and network SNR thresholds, minimum detector count).
- Distance prior power, distance bounds, `rescale_loglikelihood`.
- Root random seed, worker count, output directory.

One pinned reference config (HLV, a single fixed source) ships with the package as
the validation target.

## SNR-level simulation

For each detector `j`, with reference amplitude `|h(f)|` for a face-on source at
`D_ref` and one-sided PSD `S_j`:

```text
sigma_j^2 = 4 * df * sum_band(|h|^2 / S_j)
horizon_j = D_ref * sigma_j                  # SNR = 1 distance, Mpc
A_j(tau)  = (4 * df / sigma_j^2) * sum_band(|h|^2 / S_j * exp(2*pi*i*f*tau))
```

The expected complex SNR has the form

```text
rho_j = (horizon_j / d_L) * [F+_j * (1 + cos^2 i) / 2  +  i * Fx_j * cos i]
```

This is the form in `bayestar_realize_coincs.simulate_snr`: the cross term enters
with `+i`, and there is no coalescence-phase factor (the tool treats it as
arbitrary). `F+` and `Fx` come from `lal.ComputeDetAMResponse`, so the
polarization convention is LAL's.

Signal and noise come from one inverse FFT per detector. With `w_f` a complex
standard normal draw per frequency bin (unit variance per quadrature):

```text
z_f    = (4 * df / sigma_j^2) * |h|^2 / S_j * rho_j * exp(-2*pi*i*f*t_j)
       + (2 * sqrt(df) / sigma_j) * |h| / sqrt(S_j) * w_f
z_j(t) = sum_band(z_f * exp(2*pi*i*f*t))
```

- The arrival time `t_j = t_geo - n . r_j / c` enters as a phase ramp, so
  sub-sample shifts are exact and need no interpolation.
- Populate positive-frequency bins of a full complex array and use `jnp.fft.ifft`
  with its `1/N` accounted for. Zero bins outside the band.
- The window `1/df` must be long enough that `A_j` has decayed at the edges; a low
  `f_low` gives long correlation tails. Test convergence, then crop to the
  light-travel time across the network plus a margin, as `bayestar.condition`
  does.
- Derive each event's key as `fold_in(root_key, event_id)`, so results do not
  depend on batch composition or order.
- Compute detection flags inside JIT and select detected events on the host.
- Enable x64 in the application. Keep times as offsets from a reference epoch.

## BAYESTAR numerical interface

```python
from ligo.skymap import core

pixels, log_bci, log_bsn = core.toa_phoa_snr(
    min_distance=dmin_mpc,
    max_distance=dmax_mpc,
    prior_distance_power=2,
    cosmology=False,
    gmst=gmst,
    sample_rate=sample_rate,
    epochs=epochs,
    snrs=snrs,
    responses=responses,
    locations=locations,
    horizons=horizons,
    rescale_loglikelihood=likelihood_scale,
)
```

| Argument | Meaning |
| --- | --- |
| `gmst` | Sidereal angle in radians at the common reference epoch |
| `sample_rate` | Common SNR sampling frequency in Hz |
| `epochs` | Per-detector first-sample time offsets from that epoch, seconds |
| `snrs` | `(n_detectors, n_samples, 2)`: amplitude and unwrapped phase |
| `responses` | `(n_detectors, 3, 3)` Earth-fixed detector response tensors |
| `locations` | `(n_detectors, 3)` positions in light-seconds, centred on the timing-weighted mean as the wrapper does |
| `horizons` | `(n_detectors,)` SNR=1 distances in Mpc |

- Reproduce the conditioning in `bayestar.condition` and the wrapper's network
  centring; shift geometry and times together.
- The C layer casts SNRs and responses to float32 and the rest to float64.
- The high-level wrapper defaults to `rescale_loglikelihood=0.83`, a calibration
  for search triggers. For known-template injections compare against `1.0` and
  choose by coverage.
- The default distance prior is `d_L^2`. The cosmological prior path is marked
  unreviewed upstream, so the high-redshift prior is an open choice.
- Run one event per process with `OMP_NUM_THREADS=1`, or few processes with more
  threads, never both.

## High-SNR behaviour (go/no-go)

This is the main technical risk for ET and CE. The distance integral is not
rescaled to the event's SNR. It is read from a 400 x 400 lookup table built per
call, whose amplitude range is set by the detector horizons; the peak of the
likelihood is factored out analytically before interpolation, so the table holds a
smooth function. Whether that table stays accurate at SNR in the hundreds is
untested and is part of this gate. The sky sampling does not adapt to SNR either:

- **Coarse first pass.** The search starts at nside 16 (3072 pixels) and, at each
  of 7 levels, subdivides the 768 highest-probability pixels (confirmed). At SNR in the hundreds the timing
  precision is tens of microseconds while a coarse pixel centre is off by about a
  millisecond, so the first pass ranks pixels on far-tail likelihoods where
  autocorrelation sidelobes can mislead the refinement.
- **SNR-series interpolation.** Log-likelihood errors scale as SNR squared times
  the interpolation error, so the sample rate that suffices at SNR 10 may not at
  SNR 300. Raising the sample rate of the crops is cheap here.
- **Resolution floor.** The maximum is nside 2048 (confirmed), so the finest pixel
  is about 8e-4 deg^2. Every map has 19200 pixels, and at most 3072 of them are at
  the finest level, covering about 2.5 deg^2. This limits only the best-localised
  events.

Status (zero-noise probe, one source; details in [audit.md](audit.md)): sky area
scales as SNR^-2 up to 1000, and the SNR-series sample rate must be raised to
16384 Hz for the sky position to be unbiased there. The PyPI wheel of
ligo.skymap 2.5.4 (macOS arm64) fails to evaluate the per-pixel distance moments
for part of the sky probability from about SNR 400 (8%), reaching 52% at 1000.
A clang build of the same source does not, and the table size is not the cause,
so this is a build-specific defect, not a limit of BAYESTAR. injstar excludes the
failed pixels and reports `distance_lost_prob`, so affected events are visible.
Still to do: check the Linux wheels, and measure the share of sources above
SNR 400 in a realistic population.

Test: one source in ET+2CE at network SNR 10, 30, 100, 300 and 1000, many noise
realisations each. Pass if `A90` scales as SNR^-2 and the truth falls inside
credible regions at the nominal rate. If it fails, try higher sample rates first;
otherwise loud events need a different sampler and the package scope changes.

## Outputs

Keep every BAYESTAR product so later projects are not limited by today's choices.

- **Per-event multi-order FITS**, written with `ligo.skymap.io.write_sky_map`:
  `UNIQ`, `PROBDENSITY`, `DISTMU`, `DISTSIGMA`, `DISTNORM` and the standard
  header. Follow `bayestar.localize` postprocessing for the distance moments.
- **Summary table**, one row per injection including non-detections: truth,
  per-detector and network SNR, detection flag, `A50`/`A90`, searched area and
  probability, marginal distance quantiles, `log_bci`, `log_bsn`, FITS path.
- **Run manifest**: the three inputs (or their hashes), root seed, package and
  dependency versions.

The marginal distance PDF is `p(d_L) = sum_i w_i * p(d_L | pixel_i)` with
`w_i = PROBDENSITY_i * pixel_area_i`; use `ligo.skymap.distance.marginal_pdf`.
`DISTMU` and `DISTSIGMA` are ansatz parameters, not the mean and standard
deviation.

Runs are resumable by event id. Measure the FITS size per event before a
population run and decide then whether to keep maps for every event.

## Package layout

```text
network    spec parsing -> response tensors, locations
psd        loading, ASD/PSD handling, resampling to the frequency grid
population file reading, unit and frame conversions
simulate   JAX SNR-level simulation and detection flags
localize   packing inputs, core.toa_phoa_snr, worker pool
products   FITS, summary table, manifest
cli        run / resume / summarise
```

Dependencies: `jax`, `ripplegw` (ripple's distribution name), `ligo.skymap`,
`astropy`. PyCBC and gwmock-signal are in the `validation` extra.

## Strain-level path (optional extra)

A strain simulation (waveform, projection, noise, matched filter) is not needed
for the baseline. Build it only to:

- confirm once that it reproduces the SNR-level statistics, and
- study model mismatch: Earth rotation, finite-arm response, higher modes,
  precession. A richer injection does not extend BAYESTAR's likelihood, so these
  are bias studies.

gwmock-signal's on-device projection kernels and ripple are the building blocks;
PyCBC is the reference for filter normalisation.

## Using the outputs in corrtracers

corrtracers currently uses a lognormal distance kernel in
`notebooks/euclid_bbh.py` and a Gaussian beam in `src/corrtracers/beam.py`. The
package output replaces neither automatically:

1. A posterior `p(d_L | data)` includes the inference prior. A distance-bin
   response is the probability of assignment to an observed bin given the true
   source, under a stated assignment rule. Estimate it from the injections; do
   not stack posteriors.
2. With the maps stored, the beam needs no `A90`-to-width convention. Per event,
   `b_ell = sum_i w_i * P_ell(cos theta_i)` with `theta_i` measured from the true
   position, averaged over the detected population. This handles multimodal maps.
3. `SkyLocalization.area_deg2` corresponds to roughly the 36% credible area of a
   Gaussian. Feeding it `A90` overstates the smoothing variance about fivefold.
4. Weight by detection and population when averaging, and test whether separate
   radial and angular responses approximate the joint one.

## Implementation sequence

Status: steps 2 and 4 are implemented (`injstar snr`, `run`, `resume`,
`summarise`), ahead of the order below. Step 1 is partly covered by tests:
geometry, antenna patterns, delays and the horizon are checked against LAL, and
a zero-noise HLV map contains the truth, but the prepared arrays have not been
compared with `bayestar_realize_coincs` plus `bayestar.localize`. Steps 3 and 5
have not been run, so no localization from this package is validated yet.

0. **Pin and audit.** Done, see [audit.md](audit.md). Fix versions of JAX, ripple
   and ligo.skymap. Confirm every claim about the ligo.skymap source, the `core.toa_phoa_snr` signature, the conditioning and
   centring conventions, and that no dependency sets x64 implicitly.
1. **HLV baseline.** Implement the SNR-level simulation for one fixed source.
   Compare prepared arrays, then maps, distances and areas, against
   `bayestar_realize_coincs` plus `bayestar.localize`, first without noise.
2. **Custom geometry.** Network spec to response tensors and locations. Check
   antenna patterns and delays for ET and CE sites, including 60-degree arms.
3. **High-SNR gate.** Run the go/no-go test above before any further work.
4. **Package and batch.** File inputs, outputs, resumable worker pool. Check
   eager against JIT and single against batched results. Time simulation and
   BAYESTAR separately.
5. **Statistical calibration.** Injection ensembles for sky and distance
   coverage; choose `rescale_loglikelihood` and priors.
6. **Consumers.** corrtracers reader and response tables. Strain-level path if a
   mismatch study is wanted.

## Open questions

- Which distance prior at high redshift, and how do lensing and population priors
  enter?
- Is independent noise acceptable for ET's co-located interferometers?
- Are the signals short enough that the static response holds for the masses and
  `f_low` of interest?
- Is a map needed for every event, or a summary plus a subsample of maps?

## Sources

- [BAYESTAR simulation tutorial](https://lscsoft.docs.ligo.org/ligo.skymap/quickstart/bayestar-injections.html)
- [Documented event adapters](https://lscsoft.docs.ligo.org/ligo.skymap/io/events.html)
- [BAYESTAR conditioning, core call and postprocessing](https://github.com/lpsinger/ligo.skymap/blob/main/ligo/skymap/bayestar/__init__.py)
- [Numerical core argument parsing](https://github.com/lpsinger/ligo.skymap/blob/main/src/core.c)
- [BAYESTAR trigger simulator](https://github.com/lpsinger/ligo.skymap/blob/main/ligo/skymap/tool/bayestar_realize_coincs.py)
- [Distance PDF/CDF utilities](https://github.com/lpsinger/ligo.skymap/blob/main/ligo/skymap/distance.py)
- [BAYESTAR method paper](https://arxiv.org/abs/1508.03634)
- [PyCBC matched filtering and normalization](https://pycbc.org/pycbc/latest/html/_modules/pycbc/filter/matchedfilter.html)
- [gwmock-signal JAX detector projection](https://github.com/Leuven-Gravity-Institute/gwmock-signal/blob/main/src/gwmock_signal/projection/jax_projection.py)
- [ripple JAX waveform interface](https://gw-jax-team.github.io/ripple/latest/quickstart/)

Sources were inspected during planning on 2026-11-12. GitHub `main` links are
moving references; record commit hashes in step 0.
