"""Retrieval: index chunks and find the most relevant ones for a query.

A :class:`Retriever` composes an :class:`~industryiq.core.embeddings.Embedder`
with a :class:`~industryiq.core.vectorstore.VectorStore`. It is provider-agnostic:
swap in a real embedder or store and the retrieval logic is unchanged.
"""

import hashlib
import uuid
from typing import Any

from industryiq.core.embeddings import Embedder
from industryiq.core.vectorstore import Hit, VectorStore


def _content_hash(text: str) -> str:
    """A short, stable digest of a chunk's text, for de-duplication."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


class Retriever:
    """Embed-and-search retriever over a vector store."""

    def __init__(self, embedder: Embedder, store: VectorStore) -> None:
        self._embedder = embedder
        self._store = store

    def index(
        self,
        texts: list[str],
        metadatas: list[dict[str, Any]] | None = None,
        ids: list[str] | None = None,
    ) -> list[str]:
        """Embed ``texts`` and add them to the store.

        Each chunk gets derived metadata: ``"text"`` (the content, for
        generation), ``"chunk_index"`` (its position in this call's sequence --
        callers pass one document's chunks in order, so it is the position within
        that document, for context-window expansion), and ``"content_hash"`` (for
        de-duplication). Caller-supplied keys override these.

        Returns:
            The ids assigned to the indexed texts (generated if not supplied).
        """
        if not texts:
            return []
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in texts]
        supplied = metadatas if metadatas is not None else [{} for _ in texts]
        metadatas = [
            {"text": text, "chunk_index": index, "content_hash": _content_hash(text), **meta}
            for index, (text, meta) in enumerate(zip(texts, supplied, strict=True))
        ]
        vectors = self._embedder.embed(texts)
        self._store.upsert(ids, vectors, metadatas)
        return ids

    def retrieve(self, query: str, k: int = 5) -> list[Hit]:
        """Return up to ``k`` chunks most relevant to ``query``.

        The raw ``query`` is passed through as ``query_text`` so stores that
        support it (Milvus) can run a hybrid dense + BM25 search; dense-only
        stores ignore it.
        """
        query_vector = self._embedder.embed([query])[0]
        return self._store.search(query_vector, k=k, query_text=query)

    def all_chunks(self, limit: int = 100) -> list[tuple[str, dict[str, Any]]]:
        """Return up to ``limit`` indexed ``(id, metadata)`` pairs, for inspection."""
        return self._store.all_items(limit=limit)

    def delete_source(self, source: str) -> int:
        """Delete every indexed chunk that came from ``source``. Returns the count."""
        return self._store.delete_by_source(source)
