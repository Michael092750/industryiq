"""Vector store: index vectors and find the nearest ones to a query.

Defines the :class:`VectorStore` interface, a :class:`Hit` result type, and an
:class:`InMemoryVectorStore` for tests. The real backend (pgvector on Postgres)
is added in a later phase and must satisfy the same interface and pass the same
tests.
"""

import math
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class Hit:
    """A single search result."""

    id: str
    score: float
    metadata: dict[str, Any]


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
