"""Ephemeral per-conversation document index (the "chat on docs" side).

Holds, per conversation id, an in-memory :class:`~industryiq.core.retrieval.Retriever`
over the files dragged into that session. Nothing is persisted -- on restart the
session docs are gone, which is exactly the "memory only, valid for the session"
guarantee. The embedder is shared with the global store, so similarity scores are
comparable when the two result sets are merged in :class:`ChatService`.
"""

from industryiq.core.chat.ports import SessionDocumentStore
from industryiq.core.chunking import chunk_text
from industryiq.core.embeddings import Embedder
from industryiq.core.retrieval import Retriever
from industryiq.core.vectorstore import Hit, InMemoryVectorStore


class SessionDocuments(SessionDocumentStore):
    """In-memory, per-conversation document index."""

    def __init__(self, embedder: Embedder, *, chunk_size: int = 200, overlap: int = 20) -> None:
        self._embedder = embedder
        self._chunk_size = chunk_size
        self._overlap = overlap
        self._sessions: dict[str, Retriever] = {}

    def _session(self, conversation_id: str) -> Retriever:
        return self._sessions.setdefault(
            conversation_id, Retriever(self._embedder, InMemoryVectorStore())
        )

    def add(self, conversation_id: str, filename: str, text: str) -> list[str]:
        chunks = chunk_text(text, chunk_size=self._chunk_size, overlap=self._overlap)
        metadatas = [{"source": filename} for _ in chunks]
        return self._session(conversation_id).index(chunks, metadatas=metadatas)

    def retrieve(self, conversation_id: str, query: str, k: int = 5) -> list[Hit]:
        retriever = self._sessions.get(conversation_id)
        return retriever.retrieve(query, k=k) if retriever else []

    def documents(self, conversation_id: str) -> list[str]:
        retriever = self._sessions.get(conversation_id)
        if retriever is None:
            return []
        seen: list[str] = []
        for _id, metadata in retriever.all_chunks(limit=100000):
            source = metadata.get("source")
            if source and source not in seen:
                seen.append(source)
        return seen

    def clear(self, conversation_id: str) -> None:
        """Drop a conversation's uploaded documents (called when it is deleted)."""
        self._sessions.pop(conversation_id, None)
