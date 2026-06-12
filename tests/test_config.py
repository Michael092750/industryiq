import pytest

from ragproject.config import get_settings


def test_debug_api_key_read_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEBUG_API_KEY", "abc123")
    assert get_settings().debug_api_key == "abc123"


def test_debug_api_key_defaults_to_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DEBUG_API_KEY", raising=False)
    assert get_settings().debug_api_key is None


def test_database_url_read_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@localhost:5432/db")
    assert get_settings().database_url == "postgresql://u:p@localhost:5432/db"
