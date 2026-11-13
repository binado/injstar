import pytest

from injstar import __version__
from injstar.cli import main


def test_version(capsys: pytest.CaptureFixture[str]) -> None:
    assert __version__ == "0.1.0"
    assert main(["--version"]) == 0
    assert capsys.readouterr().out.strip() == "injstar 0.1.0"


def test_help_lists_subcommands(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--help"]) == 0

    out = capsys.readouterr().out
    for command in ("snr", "run", "summarise"):
        assert command in out
