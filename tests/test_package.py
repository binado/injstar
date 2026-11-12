from injstar import __version__


def test_version() -> None:
    assert __version__ == "0.1.0"


def test_main(capsys) -> None:
    from injstar.__main__ import main

    main()
    assert capsys.readouterr().out == "Hello from injstar!\n"
