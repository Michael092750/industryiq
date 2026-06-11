"""Retrieval: index chunks and find the most relevant ones for a query.

A :class:`Retriever` composes an :class:`~ragproject.core.embeddings.Embedder`
with a :class:`~ragproject.core.vectorstore.VectorStore`. It is provider-agnostic:
swap in a real embedder or store and the retrieval logic is unchanged.
"""

import uuid
from typing import Any

from ragproject.core.embeddings import Embedder
from ragproject.core.vectorstore import Hit, VectorStore


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

        Each chunk's text is stored in its metadata under ``"text"`` (unless a
        caller-supplied metadata already provides it) so retrieved hits carry the
        content needed for generation.

        Returns:
            The ids assigned to the indexed texts (generated if not supplied).
        """
        if not texts:
            return []
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in texts]
        if metadatas is None:
            metadatas = [{"text": text} for text in texts]
        else:
            metadatas = [
                {"text": text, **meta} for text, meta in zip(texts, metadatas, strict=True)
            ]
        vectors = self._embedder.embed(texts)
        self._store.upsert(ids, vectors, metadatas)
        return ids

    def retrieve(self, query: str, k: int = 5) -> list[Hit]:
        """Return up to ``k`` chunks most relevant to ``query``."""
        query_vector = self._embedder.embed([query])[0]
        return self._store.search(query_vector, k=k)

    def all_chunks(self, limit: int = 100) -> list[tuple[str, dict[str, Any]]]:
        """Return up to ``limit`` indexed ``(id, metadata)`` pairs, for inspection."""
        return self._store.all_items(limit=limit)
