"""Reranking adapters: implementations of the :class:`Reranker` port.

Stage 2 of a two-stage retrieve->rerank pipeline. Stage 1 (hybrid dense + BM25,
RRF-fused) fetches a *wide* candidate pool cheaply for recall; a reranker here reads
each candidate's text and re-sorts by true relevance, returning the top ``k``.

* :class:`NoOpReranker` -- identity (truncate to ``k``); the offline default and test
  double, so the pipeline shape is unchanged when no reranker is configured.
* :class:`CrossEncoderReranker` -- a local fastembed cross-encoder scoring
  ``(query, chunk)`` pairs. CPU-only, no API calls -- the reranking counterpart to the
  local (fastembed) dense embedder, and the *cheaper* alternative to the LLM strategy
  router (a local ~50-100 ms scoring pass vs. the router's ~1.4 s API round-trip).
"""

from collections.abc import Iterable
from dataclasses import replace
from typing import Protocol

from industryiq.core.retrieval.ports import Reranker
from industryiq.core.vectorstore import Hit, ScoreKind

# ms-marco-MiniLM-L-6-v2: a small, fast cross-encoder trained for passage reranking;
# the reranking counterpart to the bge-small dense embedder (both CPU ONNX via
# fastembed, no network after the first model download).
_DEFAULT_MODEL = "Xenova/ms-marco-MiniLM-L-6-v2"


class NoOpReranker(Reranker):
    """Identity reranker: keep the incoming order, truncate to ``k``.

    The default when no reranker is wired -- the two-stage pipeline collapses back to
    plain Stage-1 retrieval, so behaviour (and ordering) is unchanged.
    """

    def rerank(self, query: str, hits: list[Hit], k: int) -> list[Hit]:
        return hits[:k]


class _CrossEncoder(Protocol):
    """The slice of fastembed's ``TextCrossEncoder`` this adapter uses.

    Declaring it as a ``Protocol`` keeps the heavy fastembed model *injectable*, so
    tests pass a deterministic fake and run offline -- fastembed is the optional
    ``local`` extra, not installed in the default (offline) test/CI environment.
    """

    def rerank(self, query: str, documents: Iterable[str]) -> Iterable[float]: ...


class CrossEncoderReranker(Reranker):
    """Rerank candidates with a local cross-encoder that reads ``(query, chunk)`` pairs.

    Scores every candidate's text against the query with a fastembed
    ``TextCrossEncoder`` (a small MS-MARCO MiniLM by default), then returns the ``k``
    highest-scoring hits in descending relevance. Each returned hit carries the
    cross-encoder score tagged :attr:`~industryiq.core.vectorstore.ScoreKind.RERANK` --
    an unbounded, query-specific relevance logit (not a cosine), so a downstream cosine
    threshold never mis-fires on it (see
    :class:`~industryiq.core.retrieval.adapters.filtering.ThresholdFilter`).

    The model is CPU-only and makes no API calls, so it is the cheaper alternative to
    the LLM strategy router. fastembed is the optional ``local`` extra, imported lazily;
    pass ``encoder`` (a prebuilt model or a test double) to avoid the import entirely.
    """

    def __init__(
        self,
        model_name: str = _DEFAULT_MODEL,
        *,
        encoder: _CrossEncoder | None = None,
    ) -> None:
        if encoder is not None:
            self._encoder: _CrossEncoder = encoder
        else:
            # Lazy import: only the local extra installs fastembed (mirrors LocalEmbedder).
            from fastembed.rerank.cross_encoder import TextCrossEncoder

            self._encoder = TextCrossEncoder(model_name=model_name)

    def rerank(self, query: str, hits: list[Hit], k: int) -> list[Hit]:
        if k <= 0:
            raise ValueError("k must be positive")
        if not hits:
            return []
        texts = [hit.metadata.get("text", "") for hit in hits]
        scores = list(self._encoder.rerank(query, texts))
        ranked = sorted(zip(hits, scores, strict=True), key=lambda pair: pair[1], reverse=True)
        # Replace the Stage-1 score with the cross-encoder's, tagged RERANK so the
        # relevance filter reads it on the right (non-cosine) scale; id + metadata (the
        # chunk text) are preserved unchanged -- only the ordering and score change.
        return [
            replace(hit, score=float(score), score_kind=ScoreKind.RERANK)
            for hit, score in ranked[:k]
        ]
