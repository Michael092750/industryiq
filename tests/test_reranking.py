"""Unit tests for the reranking adapters (Stage 2 of a retrieve->rerank pipeline).

Offline: :class:`CrossEncoderReranker` is exercised with an injected fake cross-encoder
(no fastembed model download, no network), so the reorder / tag / top-k contract is
tested deterministically -- fastembed is the optional ``local`` extra, absent in CI.
"""

from collections.abc import Iterable

from industryiq.core.retrieval.adapters.reranking import CrossEncoderReranker, NoOpReranker
from industryiq.core.vectorstore import Hit, ScoreKind


def _hit(id_: str, text: str, score: float = 0.5) -> Hit:
    return Hit(id=id_, score=score, metadata={"text": text})


class FakeEncoder:
    """A cross-encoder double: scores each document by a lookup on its text."""

    def __init__(self, scores_by_text: dict[str, float]) -> None:
        self._scores = scores_by_text
        self.queries: list[str] = []

    def rerank(self, query: str, documents: Iterable[str]) -> Iterable[float]:
        self.queries.append(query)
        return [self._scores[doc] for doc in documents]


# --- NoOpReranker -------------------------------------------------------------


def test_noop_reranker_keeps_order_and_truncates() -> None:
    hits = [_hit("a", "A"), _hit("b", "B"), _hit("c", "C")]
    assert [h.id for h in NoOpReranker().rerank("q", hits, 2)] == ["a", "b"]


def test_noop_reranker_leaves_score_kind_untouched() -> None:
    [kept] = NoOpReranker().rerank("q", [_hit("a", "A")], 1)
    assert kept.score_kind is ScoreKind.COSINE  # identity: no re-tagging


# --- CrossEncoderReranker -----------------------------------------------------


def test_cross_encoder_reorders_by_score() -> None:
    hits = [_hit("a", "A"), _hit("b", "B"), _hit("c", "C")]
    enc = FakeEncoder({"A": 0.1, "B": 0.9, "C": 0.5})
    ranked = CrossEncoderReranker(encoder=enc).rerank("q", hits, 3)
    assert [h.id for h in ranked] == ["b", "c", "a"]  # rescued past the Stage-1 order


def test_cross_encoder_returns_only_top_k() -> None:
    hits = [_hit("a", "A"), _hit("b", "B"), _hit("c", "C")]
    enc = FakeEncoder({"A": 0.1, "B": 0.9, "C": 0.5})
    ranked = CrossEncoderReranker(encoder=enc).rerank("q", hits, 1)
    assert [h.id for h in ranked] == ["b"]


def test_cross_encoder_tags_rerank_score_kind_and_value() -> None:
    enc = FakeEncoder({"A": 4.2})
    [hit] = CrossEncoderReranker(encoder=enc).rerank("q", [_hit("a", "A", score=0.9)], 1)
    assert hit.score_kind is ScoreKind.RERANK  # not a cosine -> so filters don't mis-fire
    assert hit.score == 4.2  # the cross-encoder score replaces the Stage-1 score


def test_cross_encoder_preserves_id_and_metadata() -> None:
    enc = FakeEncoder({"A": 1.0})
    hit = Hit(id="a", score=0.5, metadata={"text": "A", "source": "doc.pdf"})
    [ranked] = CrossEncoderReranker(encoder=enc).rerank("q", [hit], 1)
    assert ranked.id == "a"
    assert ranked.metadata == {"text": "A", "source": "doc.pdf"}


def test_cross_encoder_empty_hits_short_circuits() -> None:
    enc = FakeEncoder({})
    assert CrossEncoderReranker(encoder=enc).rerank("q", [], 5) == []
    assert enc.queries == []  # no model call when there is nothing to score
