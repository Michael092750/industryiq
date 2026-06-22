import pytest

from industryiq.api.deps import get_chat_service, get_session_documents
from industryiq.core.chat import ChatService


def _clear_caches() -> None:
    get_chat_service.cache_clear()
    get_session_documents.cache_clear()


def test_get_chat_service_uses_in_memory_without_database_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RAG_PROVIDER", "fake")
    monkeypatch.delenv("DATABASE_URL", raising=False)
    _clear_caches()
    assert isinstance(get_chat_service(), ChatService)
    _clear_caches()


def test_get_chat_service_uses_postgres_when_database_url_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorded: dict[str, object] = {}

    class FakePgVector:
        def __init__(self, dsn: str, dim: int) -> None:
            recorded["vector_dsn"] = dsn

    class FakePgConversation:
        def __init__(self, dsn: str) -> None:
            recorded["conversation_dsn"] = dsn

    monkeypatch.setenv("RAG_PROVIDER", "fake")
    monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@host/db")
    monkeypatch.setattr("industryiq.api.deps.PgVectorStore", FakePgVector)
    monkeypatch.setattr("industryiq.api.deps.PgConversationStore", FakePgConversation)
    _clear_caches()
    assert isinstance(get_chat_service(), ChatService)
    assert recorded["vector_dsn"] == "postgresql://u:p@host/db"
    assert recorded["conversation_dsn"] == "postgresql://u:p@host/db"
    _clear_caches()
