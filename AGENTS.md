# AGENTS.md

This file provides guidance to AI agents when working with code in this repository.

## Project

`injstar` simulates BAYESTAR sky–distance posteriors for arbitrary detector networks (including ET and CE). Input: a population of compact binaries (CSV/HDF5, gwmock-pop layout) plus stage TOMLs. Output: one multi-order sky map per detected event and a summary table.

## Commands

Uses `uv` + `just` (`just setup` = `uv sync --all-groups`).

```console
just check                     # lint, fmt-check, ty, full tests, build, uv lock --check (what CI runs)
just fmt                       # ruff check --fix + ruff format
just typecheck                 # ty check (covers src and tests)
just test-fast                 # pytest -m "not slow" (skips BAYESTAR tests, which take ~1 min)
uv run pytest tests/test_cli.py::test_name   # single test
just run inject|localize ...        # uv run injstar ...
```

A prek pre-commit hook runs ruff and `ty` on the whole tree, so every commit must be a self-consistent tree.

## Styleguide

- Use type annotations
- Use numpy-style docstrings
- Follow jax design patterns
- Don't enforce patterns or invariants through comments.
Comments are for explaining non-trivial steps or gotchas

## Gotchas

- `core.toa_phoa_snr` requires `snrs` and `responses` to already be float32 (float64 raises `TypeError`).
- ligo.skymap, lal and lalsimulation are not statically analyzable; `ty` is configured with `replace-imports-with-any` for them in `pyproject.toml`.
- Packaged H1/L1/V1/K1 geodetic response tensors agree with LAL's float32 tensors. LAL arm altitudes are already radians; do not convert them from degrees. Horizon uses a rectangle sum rather than ligo.skymap's trapezoid. BAYESTAR's conditioned template can extend below the requested low-frequency cutoff; reference comparisons must match the frequency band and account for endpoint weighting.
- `inclination` is the angle between L (at `reference_frequency`) and the line of sight (LAL/ripple `iota`), not θ_JN, for precessing signals. BAYESTAR's signal model is `0.5(1+u²)Re − i·u·Im`; ripple gives `h× = −i cos ι · h₊`, so injecting ripple's strain matches it.
- The template (`template_approximant`) must be a dominant-mode aligned-spin model; the injected signal (`waveform_approximant`) may be anything ripple has.

## Git

Conventional Commits; never commit to `main` (branch first, add files by name). Release: push a `vX.Y.Z` tag matching `pyproject.toml` (PyPI), or run the Release workflow manually for TestPyPI.
