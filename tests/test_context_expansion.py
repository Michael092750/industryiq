"""Unit tests for neighbour/context expansion -- NeighborExpander over an in-memory store.

Vectors are irrelevant here (expansion uses ``fetch_neighbors``, a metadata lookup on
``source`` + ``chunk_index``), so they're arbitrary; the assertions are all on the
stitched ``text`` and on citation fields staying put.
"""

from industryiq.core.retrieval.adapters.expansion import NeighborExpander, NoOpExpander
from industryiq.core.vectorstore import Hit, InMemoryVectorStore


def _store(skip: set[int] | None = None) -> InMemoryVectorStore:
    """A 4-chunk document a.pdf (indices 0..3), two sections; optionally omit some."""
    skip = skip or set()
    rows = [
        (0, "A0", "Intro"),
        (1, "A1", "Intro"),
        (2, "A2", "Body"),
        (3, "A3", "Body"),
    ]
    rows = [row for row in rows if row[0] not in skip]
    store = InMemoryVectorStore()
    store.upsert(
        ids=[f"a{i}" for i, _, _ in rows],
        vectors=[[1.0, 0.0] for _ in rows],
        metadatas=[
            {"text": text, "source": "a.pdf", "chunk_index": i, "section": section}
            for i, text, section in rows
        ],
    )
    return store


def _hit(index: int, *, score: float = 0.9, section: str = "Intro") -> Hit:
    return Hit(
        id=f"a{index}",
        score=score,
        metadata={"text": f"A{index}", "source": "a.pdf", "chunk_index": index, "section": section},
    )


def test_expands_hit_with_its_neighbours() -> None:
    out = NeighborExpander(_store(), radius=1, max_chunks=None).expand([_hit(1)])
    assert out[0].metadata["text"] == "A0\n\nA1\n\nA2"


def test_expansion_preserves_citation_fields() -> None:
    # Only text grows; id/score/score_kind stay the matched chunk's.
    hit = _hit(1, score=0.77)
    out = NeighborExpander(_store(), radius=1).expand([hit])
    assert out[0].id == "a1"
    assert out[0].score == 0.77
    assert out[0].score_kind == hit.score_kind


def test_noop_expander_is_identity() -> None:
    hits = [_hit(1)]
    assert NoOpExpander().expand(hits) is hits


def test_radius_zero_returns_hits_unchanged() -> None:
    hits = [_hit(1)]
    assert NeighborExpander(_store(), radius=0).expand(hits) is hits


def test_absorbed_hit_is_dropped_and_text_not_duplicated() -> None:
    # Hits at 1 and 2 (radius 1): the higher-ranked hit-1 claims {0,1,2}; hit-2's
    # center is consumed, so it's dropped -- one merged span, no duplicate chunk.
    out = NeighborExpander(_store(), radius=1, max_chunks=None).expand(
        [_hit(1), _hit(2, score=0.8)]
    )
    assert [h.id for h in out] == ["a1"]
    assert out[0].metadata["text"] == "A0\n\nA1\n\nA2"


def test_partial_overlap_does_not_duplicate_shared_chunk() -> None:
    # Hits at 0 and 2 (radius 1): windows {0,1} and {1,2,3} share chunk 1. hit-0
    # claims it, so hit-2 skips it -- both survive, chunk 1 appears once.
    out = NeighborExpander(_store(), radius=1, max_chunks=None).expand(
        [_hit(0), _hit(2, score=0.8)]
    )
    assert [h.id for h in out] == ["a0", "a2"]
    assert out[0].metadata["text"] == "A0\n\nA1"
    assert out[1].metadata["text"] == "A2\n\nA3"


def test_hit_without_source_or_index_passes_through() -> None:
    lonely = Hit("x", 0.5, {"text": "lonely"})
    assert NeighborExpander(_store(), radius=1).expand([lonely]) == [lonely]


def test_max_chunks_caps_the_window_width() -> None:
    # radius 3 would reach the whole doc; max_chunks 3 trims the effective radius to 1.
    out = NeighborExpander(_store(), radius=3, max_chunks=3).expand([_hit(1)])
    assert out[0].metadata["text"] == "A0\n\nA1\n\nA2"


def test_clamp_section_stops_at_a_section_boundary() -> None:
    # Center chunk 2 is in "Body"; chunk 1 ("Intro") is excluded, chunk 3 ("Body") kept.
    out = NeighborExpander(_store(), radius=2, max_chunks=None, clamp_section=True).expand(
        [_hit(2, section="Body")]
    )
    assert out[0].metadata["text"] == "A2\n\nA3"


def test_gap_in_the_store_stops_extension() -> None:
    # Chunk 2 is missing: expanding chunk 1 (radius 1) can't reach past the gap.
    out = NeighborExpander(_store(skip={2}), radius=1, max_chunks=None).expand([_hit(1)])
    assert out[0].metadata["text"] == "A0\n\nA1"
