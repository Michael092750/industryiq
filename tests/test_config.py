import pytest

from industryiq.config import get_settings


def test_debug_api_key_read_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEBUG_API_KEY", "abc123")
    assert get_settings().debug_api_key == "abc123"


def test_debug_api_key_defaults_to_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DEBUG_API_KEY", raising=False)
    assert get_settings().debug_api_key is None


def test_database_url_read_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@localhost:5432/db")
    assert get_settings().database_url == "postgresql://u:p@localhost:5432/db"


def test_provider_defaults_to_fake(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RAG_PROVIDER", raising=False)
    assert get_settings().provider == "fake"


def test_provider_read_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_PROVIDER", "bedrock")
    settings = get_settings()
    assert settings.provider == "bedrock"
    assert settings.bedrock_llm_model_id == "us.anthropic.claude-sonnet-4-6"
    assert settings.bedrock_embed_model_id == "amazon.titan-embed-text-v2:0"


def test_cors_origins_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CORS_ORIGINS", raising=False)
    assert get_settings().cors_origins == ("http://localhost:5173",)


def test_cors_origins_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CORS_ORIGINS", "http://a.com, http://b.com")
    assert get_settings().cors_origins == ("http://a.com", "http://b.com")


def test_anthropic_api_key_read_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    assert get_settings().anthropic_api_key == "sk-ant-test"


def test_anthropic_api_key_defaults_to_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert get_settings().anthropic_api_key is None


def test_anthropic_llm_model_id_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_LLM_MODEL_ID", raising=False)
    assert get_settings().anthropic_llm_model_id == "claude-sonnet-4-6"
