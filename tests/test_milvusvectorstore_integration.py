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


def test_promoted_fields_round_trip(store: VectorStore) -> None:
    # The promoted columns plus a residual key (page) must come back as one dict,
    # byte-identical to what went in.
    meta = {
        "text": "T",
        "source": "report.pdf",
        "section": "Introduction",
        "category": "AI",
        "publisher": "McKinsey",
        "source_type": "consultancy",
        "published_date": "2024-03-15",
        "page": 3,
    }
    store.upsert(["x"], [[1.0, 0.0]], [dict(meta)])
    assert store.search([1.0, 0.0], k=1)[0].metadata == meta


def test_hybrid_search_uses_query_text_and_reports_cosine(store: VectorStore) -> None:
    store.upsert(
        ids=["a", "b", "c"],
        vectors=[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
        metadatas=[{"text": "alpha apple"}, {"text": "beta banana"}, {"text": "gamma cherry"}],
    )
    hits = store.search([0.0, 1.0], k=3, query_text="banana")
    assert hits, "hybrid search returned no hits"
    # Dense (nearest [0,1]) and BM25 ("banana") both favor b.
    assert hits[0].id == "b"
    # Score is the dense cosine similarity, not the RRF fused score.
    assert hits[0].score == pytest.approx(1.0)
    assert all(-1.0 <= hit.score <= 1.0 for hit in hits)


def test_semantic_search_is_dense_cosine(store: VectorStore) -> None:
    from industryiq.core.milvusvectorstore import MilvusVectorStore

    assert isinstance(store, MilvusVectorStore)
    _seed(store)
    hits = store.semantic_search([0.0, 1.0], k=3)
    assert [hit.id for hit in hits] == ["b", "c", "a"]
    assert hits[0].score == pytest.approx(1.0)


def test_lexical_search_matches_terms_without_embedding(store: VectorStore) -> None:
    from industryiq.core.milvusvectorstore import MilvusVectorStore

    assert isinstance(store, MilvusVectorStore)
    store.upsert(
        ids=["a", "b", "c"],
        vectors=[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
        metadatas=[{"text": "alpha apple"}, {"text": "beta banana"}, {"text": "gamma cherry"}],
    )
    hits = store.lexical_search("banana", k=3)
    assert hits, "lexical search returned no hits"
    assert hits[0].id == "b"  # only chunk containing "banana"
    assert hits[0].score > 0.0  # raw BM25 score, not a cosine


def test_weighted_search_blends_and_returns_results(store: VectorStore) -> None:
    from industryiq.core.milvusvectorstore import MilvusVectorStore

    assert isinstance(store, MilvusVectorStore)
    store.upsert(
        ids=["a", "b", "c"],
        vectors=[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
        metadatas=[{"text": "alpha apple"}, {"text": "beta banana"}, {"text": "gamma cherry"}],
    )
    # beta>0 so BM25 ("banana") contributes; dense ([0,1]) also favors b.
    hits = store.weighted_search([0.0, 1.0], "banana", k=3, alpha=0.5, beta=0.5)
    assert hits, "weighted search returned no hits"
    assert hits[0].id == "b"
    # Score is the blended/normalized fusion score, not raw cosine; ranking is
    # non-increasing because the reported score IS what we ranked by.
    assert [h.score for h in hits] == sorted((h.score for h in hits), reverse=True)


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


def test_delete_by_source_removes_only_matching_chunks(store: VectorStore) -> None:
    store.upsert(
        ids=["a1", "a2", "b1"],
        vectors=[[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]],
        metadatas=[
            {"text": "A1", "source": "a.pdf"},
            {"text": "A2", "source": "a.pdf"},
            {"text": "B1", "source": "b.pdf"},
        ],
    )
    assert store.delete_by_source("a.pdf") == 2
    assert {id_ for id_, _meta in store.all_items()} == {"b1"}


def test_delete_by_source_unknown_source_is_noop(store: VectorStore) -> None:
    _seed(store)
    assert store.delete_by_source("missing.pdf") == 0
