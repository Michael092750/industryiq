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
