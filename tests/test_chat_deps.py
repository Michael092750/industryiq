import pytest

from ragproject.api.deps import get_chat_service
from ragproject.core.chat import ChatService


def test_get_chat_service_uses_in_memory_without_database_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RAG_PROVIDER", "fake")
    monkeypatch.delenv("DATABASE_URL", raising=False)
    get_chat_service.cache_clear()
    assert isinstance(get_chat_service(), ChatService)
    get_chat_service.cache_clear()


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
    monkeypatch.setattr("ragproject.api.deps.PgVectorStore", FakePgVector)
    monkeypatch.setattr("ragproject.api.deps.PgConversationStore", FakePgConversation)
    get_chat_service.cache_clear()
    assert isinstance(get_chat_service(), ChatService)
    assert recorded["vector_dsn"] == "postgresql://u:p@host/db"
    assert recorded["conversation_dsn"] == "postgresql://u:p@host/db"
    get_chat_service.cache_clear()
