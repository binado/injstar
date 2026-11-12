"""Command-line interface.

Defines the ``injstar`` command and its ``run``, ``resume`` and ``summarise``
subcommands. The subcommands are placeholders until the pipeline exists.

This module must stay cheap to import: JAX and ligo.skymap are imported by the
modules that use them, never here.
"""

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from injstar import __version__

NOT_IMPLEMENTED_EXIT_CODE = 2


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the ``injstar`` command."""
    parser = argparse.ArgumentParser(
        prog="injstar",
        description=(
            "Simulated sky-distance posteriors with BAYESTAR for arbitrary "
            "detector networks."
        ),
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser(
        "run", help="Simulate and localize a population of injections."
    )
    run.add_argument("config", type=Path, help="Run config file.")

    resume = subparsers.add_parser(
        "resume", help="Continue an interrupted run, skipping finished events."
    )
    resume.add_argument("output_dir", type=Path, help="Output directory of the run.")

    summarise = subparsers.add_parser(
        "summarise", help="Rebuild the summary table from a run's sky maps."
    )
    summarise.add_argument("output_dir", type=Path, help="Output directory of the run.")

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the ``injstar`` command and return its exit code."""
    args = build_parser().parse_args(argv)
    print(f"injstar {args.command}: not implemented yet", file=sys.stderr)
    return NOT_IMPLEMENTED_EXIT_CODE
