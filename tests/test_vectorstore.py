import pytest

from ragproject.core.vectorstore import InMemoryVectorStore, VectorStore, cosine_similarity


def test_cosine_of_zero_vector_is_zero() -> None:
    assert cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0


def _seeded_store() -> InMemoryVectorStore:
    store = InMemoryVectorStore()
    store.upsert(
        ids=["a", "b", "c"],
        vectors=[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
        metadatas=[{"text": "A"}, {"text": "B"}, {"text": "C"}],
    )
    return store


def test_satisfies_interface() -> None:
    assert isinstance(InMemoryVectorStore(), VectorStore)


def test_search_returns_nearest_first() -> None:
    store = _seeded_store()
    hits = store.search([0.0, 1.0], k=3)
    # query == b exactly; c is at 45 degrees; a is orthogonal.
    assert [hit.id for hit in hits] == ["b", "c", "a"]


def test_exact_match_scores_one() -> None:
    store = _seeded_store()
    top = store.search([0.0, 1.0], k=1)[0]
    assert top.id == "b"
    assert top.score == pytest.approx(1.0)


def test_search_respects_k() -> None:
    store = _seeded_store()
    assert len(store.search([0.0, 1.0], k=2)) == 2


def test_metadata_is_carried_through() -> None:
    store = _seeded_store()
    top = store.search([0.0, 1.0], k=1)[0]
    assert top.metadata == {"text": "B"}


def test_search_on_empty_store_returns_empty() -> None:
    assert InMemoryVectorStore().search([1.0, 0.0], k=5) == []


def test_upsert_replaces_existing_id() -> None:
    store = InMemoryVectorStore()
    store.upsert(["a"], [[1.0, 0.0]], [{"v": 1}])
    store.upsert(["a"], [[0.0, 1.0]], [{"v": 2}])
    top = store.search([0.0, 1.0], k=1)[0]
    assert top.metadata == {"v": 2}


def test_mismatched_lengths_raise() -> None:
    store = InMemoryVectorStore()
    with pytest.raises(ValueError):
        store.upsert(["a", "b"], [[1.0, 0.0]], [{"v": 1}])


def test_invalid_k_raises() -> None:
    with pytest.raises(ValueError):
        _seeded_store().search([1.0, 0.0], k=0)
