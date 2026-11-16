"""Command-line interface; see :mod:`injstar.cli._app` for the usage summary."""

from injstar.cli import collect_summary as collect_summary
from injstar.cli import inject as inject
from injstar.cli import localize as localize
from injstar.cli._app import app, main

__all__ = ["app", "main"]
