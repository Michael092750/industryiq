from industryiq import version


def test_version_returns_string() -> None:
    assert version() == "0.0.1"
    assert isinstance(version(), str)
