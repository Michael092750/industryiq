"""Vector store: index vectors and find the nearest ones to a query.

Defines the :class:`VectorStore` interface, a :class:`Hit` result type, and an
:class:`InMemoryVectorStore` for tests. The real backend (pgvector on Postgres)
is added in a later phase and must satisfy the same interface and pass the same
tests.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol, runtime_checkable


class ScoreKind(Enum):
    """What a :class:`Hit`'s ``score`` numerically *is*, so consumers can read it right.

    A bare float is meaningless without its scale. A cosine similarity is bounded
    ``[-1, 1]`` and calibrated *across* queries, so an absolute threshold transfers.
    A raw BM25 weight is unbounded and query-specific; a min-max--normalized blend
    is query-*relative* in ``[0, 1]`` (the worst hit of every query floors near 0).
    For those two an absolute cosine cutoff is nonsense -- it would pass every BM25
    hit and chop a fixed slice off every normalized set. Carrying the kind on the
    hit lets a :class:`~industryiq.core.chat.ports.RelevanceFilter` pick the right
    policy per kind instead of assuming cosine.
    """

    COSINE = "cosine"  # bounded, cross-query comparable -> value thresholds work
    BM25 = "bm25"  # unbounded lexical weight, query-specific -> value thresholds don't transfer
    NORMALIZED = "normalized"  # min-max blend in [0, 1], query-relative -> ditto


class SearchStrategy(Enum):
    """Which retrieval mode a :class:`StrategicSearch` store should run for a query.

    The vocabulary a strategy router picks from, matched one-to-one to a store's
    search primitives. Each mode reports a different :class:`ScoreKind`, so the
    downstream relevance filter reads it on the right scale.

    * ``SEMANTIC`` -- dense/vector only (COSINE); conceptual, paraphrased queries.
    * ``LEXICAL`` -- BM25 only (BM25); exact terms/acronyms/codes the embedder blurs.
    * ``HYBRID_RRF`` -- dense + BM25, rank-fused (COSINE); the safe default for mixed
      queries. Equivalent to today's :meth:`VectorStore.search` with a ``query_text``.
    * ``HYBRID_WEIGHTED`` -- dense + BM25, score-fused with tunable weights (NORMALIZED);
      when the dense/lexical balance should be dialled rather than rank-fused equally.
    """

    SEMANTIC = "semantic"
    LEXICAL = "lexical"
    HYBRID_RRF = "hybrid_rrf"
    HYBRID_WEIGHTED = "hybrid_weighted"


@dataclass(frozen=True)
class MetadataFilter:
    """A structured pre-filter over the promoted, server-searchable metadata columns.

    Carries *what* to constrain, not *how* -- a store compiles it into its own filter
    dialect (e.g. Milvus' boolean expression), so a strategy router can request
    "publisher = McKinsey, since 2024" without knowing any store syntax. ``None``
    fields are unconstrained. ``published_from``/``published_to`` are inclusive ISO
    date bounds (``"2024"``, ``"2024-03-15"``); the ``published_date`` column sorts
    lexicographically = chronologically, so string comparison is a real date range.
    """

    publisher: str | None = None
    source_type: str | None = None
    category: str | None = None
    published_from: str | None = None
    published_to: str | None = None

    def is_empty(self) -> bool:
        """True when no field constrains anything (nothing to filter on)."""
        return all(
            value is None
            for value in (
                self.publisher,
                self.source_type,
                self.category,
                self.published_from,
                self.published_to,
            )
        )


@dataclass(frozen=True)
class SearchPlan:
    """A resolved retrieval plan: which strategy, which pre-filter, which weights.

    The output of a strategy router and the input to :meth:`StrategicSearch.search_plan`.
    Defaults reproduce today's behaviour exactly (hybrid-RRF, no filter, no weight
    override), so :meth:`is_default` marks the plan that any store can serve through
    the plain :meth:`VectorStore.search` path -- only a non-default plan needs a
    :class:`StrategicSearch`-capable store.
    """

    strategy: SearchStrategy = SearchStrategy.HYBRID_RRF
    filter: MetadataFilter | None = None
    weights: tuple[float, float] | None = None  # (dense, sparse) for HYBRID_WEIGHTED

    def is_default(self) -> bool:
        """True when this plan is the polymorphic path every store can serve.

        Hybrid-RRF, no metadata filter, no weight override -- i.e. exactly what
        ``VectorStore.search(query, query_text=...)`` already does. A plan that fails
        this needs a :class:`StrategicSearch` store; requesting it elsewhere raises.
        """
        return (
            self.strategy is SearchStrategy.HYBRID_RRF
            and not self.has_active_filter()
            and self.weights is None
        )

    def has_active_filter(self) -> bool:
        """True when a metadata filter is set and actually constrains something."""
        return self.filter is not None and not self.filter.is_empty()

    def without_filter(self) -> "SearchPlan":
        """This plan with the metadata filter dropped (same strategy + weights).

        Used to *broaden* a search that a filter over-constrained to zero hits --
        retry on the same strategy without the (possibly wrong/unmatched) filter,
        rather than answer with no grounding.
        """
        return SearchPlan(strategy=self.strategy, filter=None, weights=self.weights)


class UnsupportedStrategyError(RuntimeError):
    """Raised when a non-default :class:`SearchPlan` is asked of a store that can't run it.

    Deliberately a hard failure rather than a silent degrade to dense: if the router
    asked for lexical/weighted/filtered retrieval and the wired store can't do it, the
    caller should know, not receive quietly-wrong results (which would also corrupt a
    backend benchmark).
    """


@dataclass(frozen=True)
class Hit:
    """A single search result.

    ``score_kind`` names the scale of ``score`` (see :class:`ScoreKind`); it
    defaults to ``COSINE`` because every dense/hybrid store reports a cosine
    similarity, so existing producers stay correct without change -- only the
    lexical/weighted Milvus paths stamp a different kind.
    """

    id: str
    score: float
    metadata: dict[str, Any]
    score_kind: ScoreKind = ScoreKind.COSINE


@runtime_checkable
class VectorStore(Protocol):
    """Anything that can store vectors and search them by similarity."""

    def upsert(
        self,
        ids: list[str],
        vectors: list[list[float]],
        metadatas: list[dict[str, Any]],
    ) -> None:
        """Insert or replace vectors keyed by ``ids``, with parallel metadata."""
        ...

    def search(self, query: list[float], k: int = 5, *, query_text: str | None = None) -> list[Hit]:
        """Return up to ``k`` hits, highest cosine similarity first.

        ``query_text`` is the raw query string, for stores that can also run a
        lexical/full-text (e.g. BM25) pass; dense-only stores ignore it.
        """
        ...

    def all_items(self, limit: int = 100) -> list[tuple[str, dict[str, Any]]]:
        """Return up to ``limit`` stored ``(id, metadata)`` pairs, for inspection."""
        ...

    def delete_by_source(self, source: str) -> int:
        """Delete every chunk whose ``metadata["source"]`` equals ``source``.

        Used to replace a document's chunks when its file changes (delete the old
        set, then re-ingest). Returns the number of chunks removed.
        """
        ...

    def fetch_neighbors(self, source: str, indices: list[int]) -> dict[int, dict[str, Any]]:
        """Return the metadata of the chunks at ``indices`` within one ``source``.

        Keyed by ``chunk_index`` so a caller can stitch a retrieved chunk together
        with its neighbours (``chunk_index`` is contiguous per source, in reading
        order -- see :meth:`industryiq.core.retrieval.retriever.Retriever.index`).
        Indices with no chunk are simply absent from the result. Used by context
        expansion to recover a fact that sits in the previous/next chunk.
        """
        ...


@runtime_checkable
class StrategicSearch(Protocol):
    """A store that can execute an explicit :class:`SearchPlan` -- multiple retrieval
    modes plus a metadata pre-filter.

    A capability layered *above* :class:`VectorStore`: only stores that actually
    implement lexical/weighted/filtered retrieval (Milvus) declare it. Dense-only
    stores (pgvector, in-memory) deliberately do not, so a non-default plan asked of
    them raises :class:`UnsupportedStrategyError` instead of degrading silently. The
    default plan never needs this -- it goes through :meth:`VectorStore.search`.
    """

    def search_plan(
        self,
        *,
        query_vector: list[float],
        query_text: str,
        k: int,
        plan: SearchPlan,
    ) -> list[Hit]: ...


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity of two equal-length vectors; 0.0 if either is all zeros."""
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


class InMemoryVectorStore(VectorStore):
    """A dict-backed vector store for tests and local development."""

    def __init__(self) -> None:
        self._vectors: dict[str, list[float]] = {}
        self._metadatas: dict[str, dict[str, Any]] = {}

    def upsert(
        self,
        ids: list[str],
        vectors: list[list[float]],
        metadatas: list[dict[str, Any]],
    ) -> None:
        if not (len(ids) == len(vectors) == len(metadatas)):
            raise ValueError("ids, vectors, and metadatas must have equal length")
        for id_, vector, metadata in zip(ids, vectors, metadatas, strict=True):
            self._vectors[id_] = vector
            self._metadatas[id_] = metadata

    def search(self, query: list[float], k: int = 5, *, query_text: str | None = None) -> list[Hit]:
        # Dense-only store: query_text is accepted for protocol parity but unused.
        if k <= 0:
            raise ValueError("k must be positive")
        hits = [
            Hit(id=id_, score=cosine_similarity(query, vector), metadata=self._metadatas[id_])
            for id_, vector in self._vectors.items()
        ]
        hits.sort(key=lambda hit: hit.score, reverse=True)
        return hits[:k]

    def all_items(self, limit: int = 100) -> list[tuple[str, dict[str, Any]]]:
        return list(self._metadatas.items())[:limit]

    def delete_by_source(self, source: str) -> int:
        ids = [id_ for id_, meta in self._metadatas.items() if meta.get("source") == source]
        for id_ in ids:
            del self._vectors[id_]
            del self._metadatas[id_]
        return len(ids)

    def fetch_neighbors(self, source: str, indices: list[int]) -> dict[int, dict[str, Any]]:
        wanted = set(indices)
        return {
            index: meta
            for meta in self._metadatas.values()
            if meta.get("source") == source and (index := meta.get("chunk_index")) in wanted
        }


class MultiVectorStore(VectorStore):
    """Fan-out vector store: writes to every backend, reads from the first.

    Wraps several :class:`VectorStore` backends so a single ingest run lands
    *identical* data in all of them -- same ids, vectors, and metadata, embedded
    only once upstream by the :class:`~industryiq.core.retrieval.Retriever`. The
    point is benchmarking: load pgvector and Milvus from one bulk-ingest so the
    two query sides can be compared against the same corpus, with one shared
    ingestion manifest (every backend sees every file, so dedup stays correct).

    Writes (``upsert``, ``delete_by_source``) fan out to all backends, in order.
    Reads (``search``, ``all_items``) go to the *primary* (the first backend)
    only, so the query path is unambiguous -- once both are loaded, benchmark a
    backend's query side by pointing the app straight at it
    (``VECTOR_BACKEND=pgvector`` / ``=milvus``); the primary here just keeps the
    app functional while in fan-out mode.

    Fan-out is not transactional: if a backend fails mid-``upsert`` the others may
    already have the chunk, leaving backends out of sync. For a clean benchmark,
    load into empty collections and re-run on failure (the manifest only commits a
    file once its fan-out fully succeeds).
    """

    def __init__(self, stores: Sequence[VectorStore]) -> None:
        if not stores:
            raise ValueError("MultiVectorStore needs at least one backend")
        self._stores: tuple[VectorStore, ...] = tuple(stores)
        self._primary = self._stores[0]

    def upsert(
        self,
        ids: list[str],
        vectors: list[list[float]],
        metadatas: list[dict[str, Any]],
    ) -> None:
        for store in self._stores:
            store.upsert(ids, vectors, metadatas)

    def search(self, query: list[float], k: int = 5, *, query_text: str | None = None) -> list[Hit]:
        return self._primary.search(query, k=k, query_text=query_text)

    def all_items(self, limit: int = 100) -> list[tuple[str, dict[str, Any]]]:
        return self._primary.all_items(limit=limit)

    def delete_by_source(self, source: str) -> int:
        # Fan out to every backend; report the primary's count (they should agree
        # when the backends are in sync).
        counts = [store.delete_by_source(source) for store in self._stores]
        return counts[0]

    def fetch_neighbors(self, source: str, indices: list[int]) -> dict[int, dict[str, Any]]:
        # Read path: primary only, like search/all_items.
        return self._primary.fetch_neighbors(source, indices)
