"""Command-line interface; see :mod:`injstar.cli._app` for the usage summary."""

from injstar.cli import run as run
from injstar.cli import snr as snr
from injstar.cli import summarise as summarise
from injstar.cli._app import app, main

__all__ = ["app", "main"]
