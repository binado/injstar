"""Command-line interface; see :mod:`injstar.cli._app` for the usage summary."""

from injstar.cli import commands as commands
from injstar.cli._app import app, main

__all__ = ["app", "main"]
