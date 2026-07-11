"""Unit tests for the Redis onboarding seam (offline -- no server needed).

Covers the connection factory, the ping liveness helper's error handling, and the
``get_redis`` provider's "None when unconfigured" contract. A real round-trip
against a live server lives in test_redis_client_integration.py.
"""

import pytest
from redis import Redis
from redis.exceptions import ConnectionError as RedisConnectionError

from industryiq.api import deps
from industryiq.core.chat.adapters.session_documents import SessionDocuments
from industryiq.core.chat.adapters.session_documents_redis import RedisSessionDocumentStore
from industryiq.core.redis_client import build_redis_client, ping


def test_build_redis_client_is_lazy_and_decodes() -> None:
    # from_url opens no socket, so this never touches the network. decode_responses
    # is what makes reads come back as str (a Redis[str]).
    client = build_redis_client("redis://localhost:6379/0")
    assert isinstance(client, Redis)
    assert client.connection_pool.connection_kwargs["decode_responses"] is True


class _StubClient:
    """Minimal stand-in exercising ping()'s two branches without a server."""

    def __init__(self, *, raises: bool) -> None:
        self._raises = raises

    def ping(self) -> bool:
        if self._raises:
            raise RedisConnectionError("unreachable")
        return True


def test_ping_true_when_server_answers() -> None:
    assert ping(_StubClient(raises=False)) is True  # type: ignore[arg-type]


def test_ping_false_on_connection_error() -> None:
    # A liveness probe reports down, it does not blow up the caller.
    assert ping(_StubClient(raises=True)) is False  # type: ignore[arg-type]


def test_get_redis_none_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("REDIS_URL", raising=False)
    deps.get_redis.cache_clear()
    try:
        assert deps.get_redis() is None
    finally:
        deps.get_redis.cache_clear()


def test_get_redis_builds_client_when_set(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379/0")
    deps.get_redis.cache_clear()
    try:
        client = deps.get_redis()
        assert isinstance(client, Redis)  # lazy: still no connection opened
    finally:
        deps.get_redis.cache_clear()


def _clear_session_doc_caches() -> None:
    deps.get_redis.cache_clear()
    deps.get_session_documents.cache_clear()


def test_get_session_documents_in_memory_without_redis(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("REDIS_URL", raising=False)
    monkeypatch.setenv("RAG_PROVIDER", "fake")
    _clear_session_doc_caches()
    try:
        assert isinstance(deps.get_session_documents(), SessionDocuments)
    finally:
        _clear_session_doc_caches()


def test_get_session_documents_uses_redis_when_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379/0")
    monkeypatch.setenv("RAG_PROVIDER", "fake")
    _clear_session_doc_caches()
    try:
        # Lazy client: selecting the Redis store opens no connection.
        assert isinstance(deps.get_session_documents(), RedisSessionDocumentStore)
    finally:
        _clear_session_doc_caches()
