# injstar

Simulated sky-distance posteriors with BAYESTAR for arbitrary detector networks.

Given a population of compact binaries and a detector network, `injstar`
simulates each detector's matched-filter SNR time series and runs BAYESTAR on
the detected events.

The pipeline runs end to end, but its statistical calibration has not been
done: no coverage test and no high-SNR validation has been run yet.

## Setup

[Install uv](https://docs.astral.sh/uv/getting-started/installation/) and
[Just](https://just.systems/man/en/packages.html), then run:

```console
just setup
uv run prek install
```

The underlying command is `uv sync --all-groups` if Just is unavailable.


## Run

A run is one TOML file. This is the reference config,
[tests/fixtures/run.toml](tests/fixtures/run.toml):

```toml
network = ["H1", "L1", "V1"]     # detector names, or a name from [networks]
population = "population.csv"    # CSV or HDF5, gwmock-pop layout
output_dir = "out"

minimum_frequency = 20.0
maximum_frequency = 1024.0

[prior]
max_distance = 5000.0            # Mpc, required
```

```console
uv run injstar snr run.toml        # SNRs and detection flags only
uv run injstar run run.toml        # also one BAYESTAR sky map per detection
uv run injstar resume out          # finish an interrupted run
uv run injstar summarise out       # rebuild summary.h5 from the sky maps
```

`snr` never calls BAYESTAR, so it is the cheap way to size a run. `run`
reuses its result. The population file needs no SNR column; any it has is
carried through as `population_<name>` and not used.

The output directory holds:

| File | Contents |
| --- | --- |
| `injections.h5` | One row per injection: truth, expected and observed SNR per detector, network SNR, `detected`. |
| `skymaps/<event_id>.multiorder.fits` | BAYESTAR sky map of each detected event. |
| `summary.h5` | `injections.h5` plus 50% and 90% areas, searched area and probability, distance quantiles, `log_bci`, `log_bsn`. |
| `config.toml`, `detectors.toml`, `manifest.json` | Resolved inputs, their hashes and software versions. |

### Population columns

`detector_frame_mass_1`, `detector_frame_mass_2` (solar masses), `distance`
(Mpc), `right_ascension`, `declination`, `inclination`, `polarization_angle`
(radians) and `coa_time` (geocentric GPS seconds). `spin_1z`, `spin_2z` and
`event_id` are optional. Source-frame masses with `redshift` are accepted
instead of detector-frame masses.

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
