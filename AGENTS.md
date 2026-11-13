# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

`injstar` simulates BAYESTAR sky–distance posteriors for arbitrary detector networks (including ET and CE). Input: a population of compact binaries (CSV/HDF5, gwmock-pop layout) plus a run TOML. Output: one multi-order sky map per detected event and a summary table. README.md covers the config schema, CLI and output files; `docs/design.md` is the design and `docs/audit.md` records checks against the ligo.skymap source. Statistical calibration (coverage test, high-SNR validation, `rescale_loglikelihood`) has not been done.

## Commands

Uses `uv` + `just` (`just setup` = `uv sync --all-groups`).

```console
just check                     # lint, fmt-check, ty, full tests, build, uv lock --check (what CI runs)
just fmt                       # ruff check --fix + ruff format
just typecheck                 # ty check (covers src and tests)
just test-fast                 # pytest -m "not slow" (skips BAYESTAR tests, which take ~1 min)
uv run pytest tests/test_cli.py::test_name   # single test
just run snr|run|resume|summarise ...        # uv run injstar ...
```

A prek pre-commit hook runs ruff and `ty` on the whole tree, so every commit must be a self-consistent tree.

## Architecture

The pipeline is in `src/injstar/pipeline.py` (`compute_snr`, `run`, `resume`, `summarise`); `cli/` is a thin typer layer over it.

1. **SNR-level simulation, not strain** (`simulate.py`, JAX, batched). The matched-filter SNR series is drawn directly in closed form (expected SNR times the template autocorrelation, plus correlated complex Gaussian noise), which is statistically identical to strain simulation and mirrors `bayestar_realize_coincs`. Detection flags come from this step. `snr` stops here and writes `injections.h5`.
2. **Localization** (`localize.py`). `pack` builds the arrays `ligo.skymap.core.toa_phoa_snr` needs (mirroring `bayestar.condition`), called once per detected event on CPU with a process pool.
3. **Products** (`products.py`): per-event `skymaps/<id>.multiorder.fits`, `summary.h5` (areas, searched area/probability, distance quantiles, `log_bci`, `log_bsn`), `manifest.json`.

Things that span files:

- **Determinism enables resume.** Simulation is deterministic per event, so `run` re-draws the SNR series of events it localizes rather than storing them. `resume` finishes an interrupted run by calling `run` again.
- **Run-directory fingerprint.** The manifest hashes the config (minus `_EXECUTION_FIELDS`: output_dir, detector_config, workers, threads_per_worker, batch_size), detectors, and population/PSD file contents. A directory is reused only if the fingerprint matches. `--overwrite` refuses a non-empty directory without a manifest.
- **Detector code** (`detector/network.py`, `detector/projection.py`) is copied from the user's astrogwb repo with the same TOML schema, plus `psd_kind = "psd" | "asd"`. Packaged geometry, sensitivity curves and a reference run live in `src/injstar/data/`. PSDs resolve through gwmock-noise (`psd.py`).
- **Runtime** (`runtime.py`): `configure_runtime()` must run before JAX initializes its backend and before libraries create CPU thread pools; it enables 64-bit JAX (tests call it in `conftest.py`).

## Gotchas

- `core.toa_phoa_snr` requires `snrs` and `responses` to already be float32 (float64 raises `TypeError`).
- ligo.skymap, lal and lalsimulation are not statically analyzable; `ty` is configured with `replace-imports-with-any` for them in `pyproject.toml`.
- Response tensors here are geodetic and differ from LAL's by up to 3e-4 (H1/L1) and 2e-3 (K1); BAYESTAR's own tools use LAL's. Horizon uses a rectangle sum rather than ligo.skymap's trapezoid.
- Tests marked `slow` call BAYESTAR; CI runs them only on Python 3.14 (matrix is 3.12–3.14).

## Git

Conventional Commits; never commit to `main` (branch first, add files by name). Release: push a `vX.Y.Z` tag matching `pyproject.toml` (PyPI), or run the Release workflow manually for TestPyPI.
