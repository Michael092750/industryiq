"""Unit tests for RedisVectorStore, run against fakeredis (no server needed).

Asserts it honors the same VectorStore contract as InMemoryVectorStore -- ranking,
validation, deletion by source -- plus its Redis-specific TTL behavior.
"""

import fakeredis
import pytest

from industryiq.core.redisvectorstore import RedisVectorStore


def _store(*, ttl_seconds: int | None = None) -> RedisVectorStore:
    return RedisVectorStore(
        fakeredis.FakeStrictRedis(decode_responses=True), "ns:test", ttl_seconds=ttl_seconds
    )


def _seed(store: RedisVectorStore) -> None:
    store.upsert(
        ids=["a", "b", "c"],
        vectors=[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
        metadatas=[{"source": "x", "text": "A"}, {"source": "y", "text": "B"}, {"source": "x"}],
    )


def test_search_ranks_by_cosine_similarity() -> None:
    store = _store()
    _seed(store)
    hits = store.search([1.0, 0.0], k=2)
    assert [hit.id for hit in hits] == ["a", "c"]  # a is identical, c is 45deg, b orthogonal
    assert hits[0].score == pytest.approx(1.0)
    assert hits[0].metadata["text"] == "A"


def test_search_caps_at_k() -> None:
    store = _store()
    _seed(store)
    assert len(store.search([1.0, 1.0], k=1)) == 1


def test_search_rejects_nonpositive_k() -> None:
    store = _store()
    _seed(store)
    with pytest.raises(ValueError):
        store.search([1.0, 0.0], k=0)


def test_search_on_empty_namespace_is_empty() -> None:
    assert _store().search([1.0, 0.0], k=5) == []


def test_upsert_rejects_length_mismatch() -> None:
    with pytest.raises(ValueError):
        _store().upsert(ids=["a"], vectors=[[1.0], [2.0]], metadatas=[{}])


def test_upsert_empty_is_noop() -> None:
    store = _store()
    store.upsert(ids=[], vectors=[], metadatas=[])
    assert store.all_items() == []


def test_upsert_replaces_by_id() -> None:
    store = _store()
    store.upsert(["a"], [[1.0, 0.0]], [{"text": "old"}])
    store.upsert(["a"], [[0.0, 1.0]], [{"text": "new"}])
    items = store.all_items()
    assert len(items) == 1
    assert items[0][1]["text"] == "new"


def test_all_items_returns_id_metadata_pairs() -> None:
    store = _store()
    _seed(store)
    assert {id_ for id_, _ in store.all_items()} == {"a", "b", "c"}


def test_delete_by_source_removes_matching_chunks() -> None:
    store = _store()
    _seed(store)
    assert store.delete_by_source("x") == 2  # a and c
    assert {id_ for id_, _ in store.all_items()} == {"b"}


def test_ttl_is_set_and_refreshed_on_upsert() -> None:
    client = fakeredis.FakeStrictRedis(decode_responses=True)
    store = RedisVectorStore(client, "ns:ttl", ttl_seconds=100)
    store.upsert(["a"], [[1.0, 0.0]], [{}])
    assert 0 < client.ttl("ns:ttl") <= 100


def test_no_ttl_leaves_key_persistent() -> None:
    client = fakeredis.FakeStrictRedis(decode_responses=True)
    store = RedisVectorStore(client, "ns:persist")
    store.upsert(["a"], [[1.0, 0.0]], [{}])
    assert client.ttl("ns:persist") == -1  # -1 == no expiry set
