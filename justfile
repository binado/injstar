default:
    @just --list

setup:
    uv sync --all-groups

lock:
    uv lock

lint:
    uv run ruff check .

fmt:
    uv run ruff check --fix .
    uv run ruff format .

fmt-check:
    uv run ruff format --check .

typecheck:
    uv run ty check

test *args:
    uv run pytest {{args}}

# Everything but the tests that call BAYESTAR (about a minute).
test-fast *args:
    uv run pytest -m "not slow" {{args}}

hooks:
    uv run prek run --all-files

build:
    uv build

check: lint fmt-check typecheck test build
    uv lock --check


run *args:
    uv run injstar {{args}}
