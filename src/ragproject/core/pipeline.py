"""Pipeline: the RAG core wired together end to end.

:class:`RagPipeline` exposes two operations:

* **ingest** -- ``load -> chunk -> index`` (add documents to the store)
* **query** -- ``retrieve -> generate`` (answer a question from the store)

It depends only on the core abstractions (a :class:`Retriever` and an
:class:`LLM`), so the whole flow runs offline with fakes in tests and against
real providers in production -- no code change.
"""

from dataclasses import dataclass
from pathlib import Path

from ragproject.core.chunking import chunk_text
from ragproject.core.generation import LLM, generate_answer
from ragproject.core.loaders import load_text
from ragproject.core.retrieval import Retriever
from ragproject.core.vectorstore import Hit


@dataclass(frozen=True)
class QueryResult:
    """The answer plus the chunks it was grounded in."""

    answer: str
    hits: list[Hit]


class RagPipeline:
    """Orchestrates ingestion and querying over a retriever + LLM."""

    def __init__(
        self,
        retriever: Retriever,
        llm: LLM,
        *,
        chunk_size: int = 200,
        overlap: int = 20,
    ) -> None:
        self._retriever = retriever
        self._llm = llm
        self._chunk_size = chunk_size
        self._overlap = overlap

    def ingest_text(self, text: str, source: str | None = None) -> list[str]:
        """Chunk ``text`` and add it to the store. Returns the chunk ids."""
        chunks = chunk_text(text, chunk_size=self._chunk_size, overlap=self._overlap)
        metadatas = [{"source": source} for _ in chunks] if source is not None else None
        return self._retriever.index(chunks, metadatas=metadatas)

    def ingest_file(self, path: str | Path) -> list[str]:
        """Load a file, chunk it, and add it to the store. Returns the chunk ids."""
        return self.ingest_text(load_text(path), source=str(path))

    def query(self, question: str, k: int = 5) -> QueryResult:
        """Retrieve relevant chunks and generate a grounded answer."""
        hits = self._retriever.retrieve(question, k=k)
        answer = generate_answer(question, hits, self._llm)
        return QueryResult(answer=answer, hits=hits)
