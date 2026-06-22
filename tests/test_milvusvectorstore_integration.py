"""Integration tests for MilvusVectorStore against a real Milvus standalone.

Run with a Milvus server available (e.g. `docker compose up -d milvus`) via:

    pytest -m integration

These are skipped from the default unit run. They assert MilvusVectorStore honors
the same VectorStore contract as InMemoryVectorStore and PgVectorStore -- the
expected orderings and scores are identical to the pgvector tests, which is what
makes the two engines directly comparable in the benchmark.
"""

import uuid
from collections.abc import Iterator

import pytest

from industryiq.config import get_settings
from industryiq.core.vectorstore import VectorStore

pytestmark = pytest.mark.integration

MILVUS_URI = get_settings().milvus_uri


@pytest.fixture
def store() -> Iterator["object"]:
    from pymilvus import MilvusClient

    from industryiq.core.milvusvectorstore import MilvusVectorStore

    collection = "test_chunks_" + uuid.uuid4().hex[:8]
    try:
        s = MilvusVectorStore(MILVUS_URI, dim=2, collection=collection)
    except Exception as exc:  # noqa: BLE001 -- no Milvus reachable -> skip, don't fail
        pytest.skip(f"Milvus not reachable at {MILVUS_URI}: {exc}")
    yield s
    MilvusClient(uri=MILVUS_URI).drop_collection(collection)


def _seed(store: VectorStore) -> None:
    store.upsert(
        ids=["a", "b", "c"],
        vectors=[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
        metadatas=[{"text": "A"}, {"text": "B"}, {"text": "C"}],
    )


def test_satisfies_interface(store: VectorStore) -> None:
    assert isinstance(store, VectorStore)


def test_search_returns_nearest_first(store: VectorStore) -> None:
    _seed(store)
    hits = store.search([0.0, 1.0], k=3)
    assert [hit.id for hit in hits] == ["b", "c", "a"]


def test_exact_match_scores_one(store: VectorStore) -> None:
    _seed(store)
    top = store.search([0.0, 1.0], k=1)[0]
    assert top.id == "b"
    assert top.score == pytest.approx(1.0)


def test_search_respects_k(store: VectorStore) -> None:
    _seed(store)
    assert len(store.search([0.0, 1.0], k=2)) == 2


def test_metadata_is_carried_through(store: VectorStore) -> None:
    _seed(store)
    assert store.search([0.0, 1.0], k=1)[0].metadata == {"text": "B"}


def test_upsert_replaces_existing_id(store: VectorStore) -> None:
    store.upsert(["a"], [[1.0, 0.0]], [{"v": 1}])
    store.upsert(["a"], [[0.0, 1.0]], [{"v": 2}])
    assert store.search([0.0, 1.0], k=1)[0].metadata == {"v": 2}


def test_all_items_lists_stored_rows(store: VectorStore) -> None:
    _seed(store)
    assert {id_ for id_, _meta in store.all_items()} == {"a", "b", "c"}


def test_search_empty_store_returns_empty(store: VectorStore) -> None:
    assert store.search([1.0, 0.0], k=5) == []


def test_invalid_k_raises(store: VectorStore) -> None:
    with pytest.raises(ValueError):
        store.search([1.0, 0.0], k=0)
