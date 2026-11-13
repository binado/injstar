# AGENTS.md

This file provides guidance to AI agents when working with code in this repository.

## Project

`injstar` simulates BAYESTAR sky–distance posteriors for arbitrary detector networks (including ET and CE). Input: a population of compact binaries (CSV/HDF5, gwmock-pop layout) plus a run TOML. Output: one multi-order sky map per detected event and a summary table.

## Commands

Uses `uv` + `just` (`just setup` = `uv sync --all-groups`).

```console
just check                     # lint, fmt-check, ty, full tests, build, uv lock --check (what CI runs)
just fmt                       # ruff check --fix + ruff format
just typecheck                 # ty check (covers src and tests)
just test-fast                 # pytest -m "not slow" (skips BAYESTAR tests, which take ~1 min)
uv run pytest tests/test_cli.py::test_name   # single test
just run snr|run|summarise ...        # uv run injstar ...
```

A prek pre-commit hook runs ruff and `ty` on the whole tree, so every commit must be a self-consistent tree.

## Styleguide

- Use type annotations
- Use numpy-style docstrings
- Follow jax design patterns
- Use pytest fixtures and `pytest.mark.parametrize` for testing
- Don't create tests that re-implement functions by hand
- Use `np.testing.assert_allclose` for array comparison
- Don't add tests that compare arrays with fine-tuned tolerances

## Gotchas

- `core.toa_phoa_snr` requires `snrs` and `responses` to already be float32 (float64 raises `TypeError`).
- ligo.skymap, lal and lalsimulation are not statically analyzable; `ty` is configured with `replace-imports-with-any` for them in `pyproject.toml`.
- Response tensors here are geodetic and differ from LAL's by up to 3e-4 (H1/L1) and 2e-3 (K1); BAYESTAR's own tools use LAL's. Horizon uses a rectangle sum rather than ligo.skymap's trapezoid.
- Tests marked `slow` call BAYESTAR; CI runs them only on Python 3.14 (matrix is 3.12–3.14).

## Git

Conventional Commits; never commit to `main` (branch first, add files by name). Release: push a `vX.Y.Z` tag matching `pyproject.toml` (PyPI), or run the Release workflow manually for TestPyPI.
