# injstar

Simulated sky-distance posteriors with BAYESTAR for arbitrary detector networks.

Given a population of compact binaries and a detector network, `injstar`
simulates each detector's matched-filter SNR time series and runs BAYESTAR on
the detected events.

The pipeline runs end to end, but its statistical calibration has not been
done: no coverage test and no high-SNR validation has been run yet.

## Setup

Python 3.13 or later is required. [Install uv](https://docs.astral.sh/uv/getting-started/installation/) and
[Just](https://just.systems/man/en/packages.html), then run:

```console
just setup
uv run prek install
```

The underlying command is `uv sync --all-groups` if Just is unavailable.


## Run

Injection and localization have separate TOML files and output directories. See
[the injection example](tests/fixtures/inject.toml) and
[the localization example](tests/fixtures/localize.toml), which share
[detector configuration](tests/fixtures/detectors.toml).

```toml
# inject.toml
network = "reference"
detector_config = ["detectors.toml"]
population = "population.csv"    # CSV or HDF5, gwmock-pop layout
output_dir = "injected"
minimum_frequency = 20.0
maximum_frequency = 1024.0
```

```toml
# localize.toml
network = "reference"
detector_config = ["detectors.toml"]
injection_dir = "injected"
output_dir = "localized"

[prior]
max_distance = 5000.0            # Mpc, required only for localization
```

```console
uv run injstar inject inject.toml
uv run injstar localize localize.toml
uv run injstar localize localize.toml --resume
uv run injstar summarise localized
```

`inject` generates waveforms, noise, SNRs and detection flags. `localize` reads
completed injection products and never generates waveforms or noise. Separate
localization directories can reuse one injection directory with different priors.
The configured detector order must match the saved detector axis; the sample
rate comes from the dataset. Population and detector source files must remain
available and consistent.

| Directory | File | Contents |
| --- | --- | --- |
| Injection | `injection_results.h5` | Complete SNRs, horizons, bandwidths, complex series, epochs and original row indices, with detector order and sample rate. |
| Injection | `injections.h5` | Population truth, per-detector and network SNR, `detected`. |
| Localization | `injections.h5` | Relative symlink to the injection catalogue. |
| Localization | `skymaps/<row_index>.multiorder.fits` | BAYESTAR maps for detected events. |
| Localization | `summary.h5` | Catalogue plus localization and distance statistics. |
| Both | `config.json` | Validated configuration for that stage, with absolute paths. |

Paths in TOML resolve relative to the TOML file; CLI path overrides resolve
relative to the working directory. A matching completed injection directory is
reused. Localization supports `--resume` to skip existing maps, or `--overwrite`
to redo its products. Overlapping stage directories are rejected, and foreign
contents are protected from deletion. Execution settings (batch size, worker
and thread counts, output directory) do not affect config matching.

Results are chunked and uncompressed. The complex128 series alone uses
`16 × injections × detectors × crop_samples` bytes. Both stages stream series
in batches of `batch_size`. The catalogues use gwmock-pop’s structured `data`
dataset and store text as UTF-8 bytes.

The catalogue symlink depends on the injection directory; moving both directories
together preserves it. `summarise` follows it to rebuild the summary. The old
`snr` and `run` commands and combined configuration are removed. Split existing
TOML files into the two stages and use separate output directories.

### Population columns

`detector_frame_mass_1`, `detector_frame_mass_2` (solar masses), `distance`
(Mpc), `right_ascension`, `declination`, `inclination`, `polarization_angle`
(radians) and `coa_time` (geocentric GPS seconds). The zero-based input row
index determines noise seeds and sky-map filenames. The spin components
`spin_1x` to `spin_2z`, the tidal deformabilities `lambda_1` and `lambda_2` and the orbital phase
`coa_phase` (radians) are optional and default to zero. Which of them a run reads depends on the waveform model
(`waveform_approximant`); a column the model does not read is ignored.
Extra columns retain their original names in both output tables. Names reserved
for generated output columns (including per-detector SNRs, detection flags,
and summary statistics) are rejected.

Convert catalogues using source-frame masses or `luminosity_distance` to the
required columns before running. Remove the obsolete
`population_cosmology` setting from run configurations. Existing run directories
based on custom event IDs must not be resumed; start a new run directory.

### Waveform models

`waveform_approximant` (default `IMRPhenomD`) names the ripple frequency-domain
model of the injected signal. It may be precessing, tidal or include higher
modes: the simulation projects the full plus and cross strain onto each
detector. `template_approximant` (default: the same model) names the
matched-filter template and the horizon, and must be a dominant-mode
aligned-spin model, because BAYESTAR's signal model is the dominant mode. Set it
when the signal is not one, for example `waveform_approximant = "IMRPhenomXPHM"`
with `template_approximant = "IMRPhenomXAS"`.

`inclination` is the angle between the orbital angular momentum (at
`reference_frequency`) and the line of sight, as in LAL and ripple. For aligned
spins this is the angle the BAYESTAR signal model uses; for a precessing signal
it is not the angle between the total angular momentum and the line of sight.

### Detectors

LIGO, Virgo, KAGRA, Einstein Telescope (triangle and 2L) and Cosmic Explorer
sites and noise curves are packaged
([geometry.toml](src/injstar/data/geometry.toml),
[sensitivity.toml](src/injstar/data/sensitivity.toml)). To add or change
detectors, list TOML files under `detector_config` in the run config:

```toml
[detectors.X1]
psd_reference = "my_asd.txt"     # packaged curve name, path, or URL
psd_kind = "asd"                 # "psd" (default) or "asd"

[detectors.X1.geometry]          # radians and metres, azimuths clockwise from North
latitude_rad = 0.76
longitude_rad = 0.18
elevation_m = 50.0
xarm_azimuth_rad = 0.34
yarm_azimuth_rad = 5.57

[networks]
mine = ["X1", "H1", "L1"]
```


## Development

```console
just fmt        # uv run ruff check --fix . && uv run ruff format .
just lint       # uv run ruff check .
just typecheck  # uv run ty check
just test       # uv run pytest
just test-fast  # skip the tests that call BAYESTAR
just hooks      # uv run prek run --all-files
just check      # full validation, tests, and package build
```

## Publishing

Configure `pypi` and `testpypi` GitHub environments with trusted publishers.
Push a tag matching the version in `pyproject.toml`, such as `v0.1.0`, to
publish to PyPI. Run the Release workflow manually to publish to TestPyPI.
