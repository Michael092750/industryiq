"""Redis-backed per-conversation document index (the shared "chat on docs" tier).

The drop-in Redis counterpart to the in-memory
:class:`~industryiq.core.retrieval.adapters.session_documents.SessionDocuments`.
Same behavior, same
:class:`~industryiq.core.retrieval.ports.SessionDocumentStore` port -- it reuses
the very same :class:`~industryiq.core.retrieval.retriever.Retriever` (chunking,
embedding, id generation, ranking), only swapping the per-conversation
:class:`~industryiq.core.vectorstore.InMemoryVectorStore` for a
:class:`~industryiq.core.redisvectorstore.RedisVectorStore` keyed by conversation.

Why this exists: the in-process dict it replaces cannot be shared across processes
and dies on restart. Backed by Redis, a session's uploaded docs are visible to
every worker/agent and survive a restart -- while still being *ephemeral* by
design: a sliding ``ttl_seconds`` (refreshed on each upload) auto-evicts idle
sessions, and deleting a conversation drops its key outright.

The store holds no per-conversation state itself -- a fresh cheap ``Retriever`` is
built per call over the conversation's Redis key -- so there is nothing in memory
to fall out of sync between processes.
"""

from typing import Any

from industryiq.core.chunking import chunk_text
from industryiq.core.embeddings import Embedder
from industryiq.core.redisvectorstore import RedisVectorStore
from industryiq.core.retrieval.ports import SessionDocumentStore
from industryiq.core.retrieval.retriever import Retriever
from industryiq.core.vectorstore import Hit

# Redis key namespace for one conversation's uploaded documents.
_KEY_PREFIX = "sessdoc:"


class RedisSessionDocumentStore(SessionDocumentStore):
    """Per-conversation document index backed by Redis (shared, TTL'd).

    ``redis`` is a ``redis.Redis`` client (``decode_responses=True``); typed
    ``Any`` for the same reason as :class:`~industryiq.core.redisvectorstore.RedisVectorStore`.
    """

    def __init__(
        self,
        redis: Any,
        embedder: Embedder,
        *,
        chunk_size: int = 200,
        overlap: int = 20,
        ttl_seconds: int | None = None,
    ) -> None:
        self._redis = redis
        self._embedder = embedder
        self._chunk_size = chunk_size
        self._overlap = overlap
        self._ttl = ttl_seconds

    def _key(self, conversation_id: str) -> str:
        return f"{_KEY_PREFIX}{conversation_id}"

    def _retriever(self, conversation_id: str) -> Retriever:
        store = RedisVectorStore(self._redis, self._key(conversation_id), ttl_seconds=self._ttl)
        return Retriever(self._embedder, store)

    def add(self, conversation_id: str, filename: str, text: str) -> list[str]:
        chunks = chunk_text(text, chunk_size=self._chunk_size, overlap=self._overlap)
        metadatas = [{"source": filename} for _ in chunks]
        return self._retriever(conversation_id).index(chunks, metadatas=metadatas)

    def retrieve(self, conversation_id: str, query: str, k: int = 5) -> list[Hit]:
        # Skip the embedding call entirely for a conversation with no uploads.
        if not self._redis.exists(self._key(conversation_id)):
            return []
        return self._retriever(conversation_id).retrieve(query, k=k)

    def documents(self, conversation_id: str) -> list[str]:
        seen: list[str] = []
        for _id, metadata in self._retriever(conversation_id).all_chunks(limit=100000):
            source = metadata.get("source")
            if source and source not in seen:
                seen.append(source)
        return seen

    def clear(self, conversation_id: str) -> None:
        """Drop a conversation's uploaded documents (called when it is deleted)."""
        self._redis.delete(self._key(conversation_id))
