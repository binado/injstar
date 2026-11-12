import pytest

from injstar import __version__
from injstar.cli import NOT_IMPLEMENTED_EXIT_CODE, main


def test_version() -> None:
    assert __version__ == "0.1.0"


def test_help_lists_subcommands(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["--help"])

    assert excinfo.value.code == 0
    out = capsys.readouterr().out
    for command in ("run", "resume", "summarise"):
        assert command in out


@pytest.mark.parametrize(
    "argv",
    [["run", "config.toml"], ["resume", "out"], ["summarise", "out"]],
)
def test_subcommands_report_not_implemented(
    argv: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(argv) == NOT_IMPLEMENTED_EXIT_CODE
    assert "not implemented yet" in capsys.readouterr().err
