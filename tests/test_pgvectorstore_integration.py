"""Integration tests for PgVectorStore against a real Postgres+pgvector.

Run with a database available (e.g. `docker compose up -d`) via:

    pytest -m integration

These are skipped from the default unit run. They assert PgVectorStore honors
the same VectorStore contract as InMemoryVectorStore.
"""

import uuid
from collections.abc import Iterator

import psycopg
import pytest

from ragproject.config import get_settings
from ragproject.core.pgvectorstore import PgVectorStore
from ragproject.core.vectorstore import VectorStore

pytestmark = pytest.mark.integration

DATABASE_URL = get_settings().database_url


@pytest.fixture
def store() -> Iterator[PgVectorStore]:
    if not DATABASE_URL:
        pytest.skip("DATABASE_URL not set")
    table = "test_chunks_" + uuid.uuid4().hex[:8]
    yield PgVectorStore(DATABASE_URL, dim=2, table=table)
    with psycopg.connect(DATABASE_URL) as conn:
        conn.execute(f"DROP TABLE IF EXISTS {table}")
        conn.commit()


def _seed(store: PgVectorStore) -> None:
    store.upsert(
        ids=["a", "b", "c"],
        vectors=[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
        metadatas=[{"text": "A"}, {"text": "B"}, {"text": "C"}],
    )


def test_satisfies_interface(store: PgVectorStore) -> None:
    assert isinstance(store, VectorStore)


def test_search_returns_nearest_first(store: PgVectorStore) -> None:
    _seed(store)
    hits = store.search([0.0, 1.0], k=3)
    assert [hit.id for hit in hits] == ["b", "c", "a"]


def test_exact_match_scores_one(store: PgVectorStore) -> None:
    _seed(store)
    top = store.search([0.0, 1.0], k=1)[0]
    assert top.id == "b"
    assert top.score == pytest.approx(1.0)


def test_search_respects_k(store: PgVectorStore) -> None:
    _seed(store)
    assert len(store.search([0.0, 1.0], k=2)) == 2


def test_metadata_is_carried_through(store: PgVectorStore) -> None:
    _seed(store)
    assert store.search([0.0, 1.0], k=1)[0].metadata == {"text": "B"}


def test_upsert_replaces_existing_id(store: PgVectorStore) -> None:
    store.upsert(["a"], [[1.0, 0.0]], [{"v": 1}])
    store.upsert(["a"], [[0.0, 1.0]], [{"v": 2}])
    assert store.search([0.0, 1.0], k=1)[0].metadata == {"v": 2}


def test_all_items_lists_stored_rows(store: PgVectorStore) -> None:
    _seed(store)
    assert {id_ for id_, _meta in store.all_items()} == {"a", "b", "c"}


def test_search_empty_store_returns_empty(store: PgVectorStore) -> None:
    assert store.search([1.0, 0.0], k=5) == []


def test_invalid_k_raises(store: PgVectorStore) -> None:
    with pytest.raises(ValueError):
        store.search([1.0, 0.0], k=0)
