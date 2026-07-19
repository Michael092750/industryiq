import pytest

from industryiq.core.vectorstore import (
    InMemoryVectorStore,
    MetadataFilter,
    MultiVectorStore,
    SearchPlan,
    SearchStrategy,
    StrategicSearch,
    VectorStore,
    cosine_similarity,
)


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


def test_all_items_returns_stored_pairs() -> None:
    items = dict(_seeded_store().all_items())
    assert set(items) == {"a", "b", "c"}
    assert items["b"] == {"text": "B"}


def test_all_items_respects_limit() -> None:
    assert len(_seeded_store().all_items(limit=2)) == 2


def test_delete_by_source_removes_only_matching_chunks() -> None:
    store = InMemoryVectorStore()
    store.upsert(
        ids=["a1", "a2", "b1"],
        vectors=[[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]],
        metadatas=[
            {"text": "A1", "source": "a.pdf"},
            {"text": "A2", "source": "a.pdf"},
            {"text": "B1", "source": "b.pdf"},
        ],
    )
    deleted = store.delete_by_source("a.pdf")
    assert deleted == 2
    assert {id_ for id_, _ in store.all_items()} == {"b1"}


def test_delete_by_source_unknown_source_is_noop() -> None:
    assert _seeded_store().delete_by_source("missing.pdf") == 0


def test_multi_store_satisfies_interface() -> None:
    assert isinstance(MultiVectorStore([InMemoryVectorStore()]), VectorStore)


def test_multi_store_rejects_no_backends() -> None:
    with pytest.raises(ValueError):
        MultiVectorStore([])


def test_multi_store_fans_writes_out_to_every_backend() -> None:
    a, b = InMemoryVectorStore(), InMemoryVectorStore()
    multi = MultiVectorStore([a, b])
    multi.upsert(["x"], [[1.0, 0.0]], [{"text": "X", "source": "x.pdf"}])
    assert dict(a.all_items()) == dict(b.all_items()) == {"x": {"text": "X", "source": "x.pdf"}}


def test_multi_store_reads_from_primary_only() -> None:
    primary, secondary = InMemoryVectorStore(), InMemoryVectorStore()
    primary.upsert(["p"], [[1.0, 0.0]], [{"text": "P"}])
    secondary.upsert(["s"], [[1.0, 0.0]], [{"text": "S"}])
    multi = MultiVectorStore([primary, secondary])
    assert [hit.id for hit in multi.search([1.0, 0.0], k=5)] == ["p"]
    assert {id_ for id_, _ in multi.all_items()} == {"p"}


def test_multi_store_delete_fans_out_and_reports_primary_count() -> None:
    a, b = InMemoryVectorStore(), InMemoryVectorStore()
    multi = MultiVectorStore([a, b])
    multi.upsert(["x"], [[1.0, 0.0]], [{"text": "X", "source": "x.pdf"}])
    assert multi.delete_by_source("x.pdf") == 1
    assert a.all_items() == b.all_items() == []


# --- SearchPlan / MetadataFilter value objects -------------------------------


def test_default_plan_is_default() -> None:
    assert SearchPlan().is_default()


@pytest.mark.parametrize(
    "plan",
    [
        SearchPlan(strategy=SearchStrategy.LEXICAL),
        SearchPlan(strategy=SearchStrategy.SEMANTIC),
        SearchPlan(weights=(0.5, 0.5)),
        SearchPlan(filter=MetadataFilter(publisher="McKinsey")),
    ],
)
def test_non_default_plans_are_not_default(plan: SearchPlan) -> None:
    assert not plan.is_default()


def test_empty_filter_keeps_plan_default() -> None:
    # A filter with no set fields constrains nothing, so the plan is still the
    # polymorphic any-store path.
    assert SearchPlan(filter=MetadataFilter()).is_default()


def test_metadata_filter_is_empty() -> None:
    assert MetadataFilter().is_empty()
    assert not MetadataFilter(published_from="2024").is_empty()


# --- StrategicSearch capability ----------------------------------------------


def test_in_memory_store_is_not_strategic() -> None:
    # Dense-only stores deliberately do not advertise StrategicSearch, so the
    # retriever's isinstance gate raises for a non-default plan instead of degrading.
    assert not isinstance(InMemoryVectorStore(), StrategicSearch)


# --- fetch_neighbors ----------------------------------------------------------


def _doc_store() -> InMemoryVectorStore:
    store = InMemoryVectorStore()
    store.upsert(
        ids=["a0", "a1", "a2", "b0"],
        vectors=[[1.0, 0.0], [0.9, 0.1], [0.8, 0.2], [0.0, 1.0]],
        metadatas=[
            {"text": "A0", "source": "a.pdf", "chunk_index": 0},
            {"text": "A1", "source": "a.pdf", "chunk_index": 1},
            {"text": "A2", "source": "a.pdf", "chunk_index": 2},
            {"text": "B0", "source": "b.pdf", "chunk_index": 0},
        ],
    )
    return store


def test_fetch_neighbors_keys_by_chunk_index_within_source() -> None:
    got = _doc_store().fetch_neighbors("a.pdf", [0, 1])
    assert set(got) == {0, 1}
    assert got[1]["text"] == "A1"


def test_fetch_neighbors_excludes_other_sources() -> None:
    # Same chunk_index in a different source must not leak in.
    got = _doc_store().fetch_neighbors("a.pdf", [0])
    assert got[0]["text"] == "A0"
    assert all(meta["source"] == "a.pdf" for meta in got.values())


def test_fetch_neighbors_omits_missing_indices() -> None:
    got = _doc_store().fetch_neighbors("a.pdf", [1, 99])
    assert set(got) == {1}


def test_fetch_neighbors_empty_indices_is_noop() -> None:
    assert _doc_store().fetch_neighbors("a.pdf", []) == {}


def test_multi_store_fetch_neighbors_reads_from_primary() -> None:
    primary, secondary = _doc_store(), InMemoryVectorStore()
    multi = MultiVectorStore([primary, secondary])
    assert multi.fetch_neighbors("a.pdf", [2])[2]["text"] == "A2"
