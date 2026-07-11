"""Unit tests for RedisSessionDocumentStore, run against fakeredis (no server).

Mirrors test_chat_session_documents.py: the Redis store must honor the same
SessionDocumentStore contract as the in-memory SessionDocuments, plus clear() and
TTL, which are its reason for existing (shared, restart-surviving, self-evicting).
"""

import fakeredis

from industryiq.core.chat.adapters.session_documents_redis import RedisSessionDocumentStore
from industryiq.core.embeddings import FakeEmbedder


def _docs(*, ttl_seconds: int | None = None) -> RedisSessionDocumentStore:
    return RedisSessionDocumentStore(
        fakeredis.FakeStrictRedis(decode_responses=True),
        FakeEmbedder(dim=16),
        ttl_seconds=ttl_seconds,
    )


def test_add_then_retrieve_finds_the_document() -> None:
    docs = _docs()
    docs.add("c1", "facts.txt", "the sky is blue")
    hits = docs.retrieve("c1", "the sky is blue", k=3)
    assert hits[0].metadata["text"] == "the sky is blue"
    assert hits[0].metadata["source"] == "facts.txt"


def test_retrieve_unknown_conversation_is_empty() -> None:
    assert _docs().retrieve("nope", "anything", k=3) == []


def test_documents_lists_uploaded_filenames() -> None:
    docs = _docs()
    docs.add("c1", "a.txt", "alpha")
    docs.add("c1", "b.txt", "beta")
    assert set(docs.documents("c1")) == {"a.txt", "b.txt"}


def test_sessions_are_isolated() -> None:
    docs = _docs()
    docs.add("c1", "a.txt", "alpha beta gamma")
    assert docs.retrieve("c2", "alpha beta gamma", k=3) == []
    assert docs.documents("c2") == []


def test_clear_removes_a_conversations_documents() -> None:
    docs = _docs()
    docs.add("c1", "a.txt", "alpha beta gamma")
    docs.clear("c1")
    assert docs.documents("c1") == []
    assert docs.retrieve("c1", "alpha beta gamma", k=3) == []


def test_add_sets_ttl_on_the_conversation_key() -> None:
    client = fakeredis.FakeStrictRedis(decode_responses=True)
    docs = RedisSessionDocumentStore(client, FakeEmbedder(dim=16), ttl_seconds=100)
    docs.add("c1", "a.txt", "alpha")
    assert 0 < client.ttl("sessdoc:c1") <= 100
