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
from typing import Any

from industryiq.core.chunking import chunk_text, split_sections
from industryiq.core.generation import LLM, generate_answer
from industryiq.core.loaders import load_pages, load_title
from industryiq.core.retrieval import Retriever
from industryiq.core.vectorstore import Hit


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

    def ingest_text(
        self,
        text: str,
        source: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> list[str]:
        """Chunk ``text`` and add it to the store. Returns the chunk ids.

        ``source`` and any extra ``metadata`` (e.g. a document ``category``) are
        attached to every chunk, so retrieved hits carry them for attribution.
        """
        chunks = chunk_text(text, chunk_size=self._chunk_size, overlap=self._overlap)
        base: dict[str, Any] = {}
        if source is not None:
            base["source"] = source
        if metadata:
            base.update(metadata)
        metadatas = [dict(base) for _ in chunks] if base else None
        return self._retriever.index(chunks, metadatas=metadatas)

    def ingest_pages(
        self,
        pages: list[str],
        *,
        source: str | None = None,
        metadata: dict[str, Any] | None = None,
        title: str | None = None,
    ) -> list[str]:
        """Chunk pre-extracted ``pages`` and add them, tagging ``page`` and ``section``.

        ``page`` (1-based) is recorded only when there is more than one page, so
        single-page / non-paginated sources don't carry a misleading number.
        Chunking is per page, so a chunk never straddles a page boundary.

        ``section`` is each chunk's nearest Markdown heading (from Docling's
        layout-aware output), carried across page boundaries so body that
        continues past a page break keeps the heading it opened under. Pages with
        no headings (plain text, pypdf) leave ``section`` unset, so chunking is
        unchanged for them.
        """
        base: dict[str, Any] = {}
        if source is not None:
            base["source"] = source
        if title:
            base["title"] = title
        if metadata:
            base.update(metadata)
        paginated = len(pages) > 1
        chunks: list[str] = []
        metadatas: list[dict[str, Any]] = []
        carried_section: str | None = None  # heading in force at the last page's end
        for page_number, page_text in enumerate(pages, start=1):
            for section, block in split_sections(page_text, initial_section=carried_section):
                carried_section = section
                for chunk in chunk_text(block, chunk_size=self._chunk_size, overlap=self._overlap):
                    chunks.append(chunk)
                    chunk_meta = dict(base)
                    if paginated:
                        chunk_meta["page"] = page_number
                    if section:
                        chunk_meta["section"] = section
                    metadatas.append(chunk_meta)
        return self._retriever.index(chunks, metadatas=metadatas)

    def ingest_file(
        self,
        path: str | Path,
        metadata: dict[str, Any] | None = None,
        source: str | None = None,
    ) -> list[str]:
        """Load a file (page-aware), chunk it, and add it to the store.

        Each chunk carries its ``page`` (for paginated formats) and the document
        ``title`` (embedded metadata, else the file name). ``source`` overrides
        the recorded path -- e.g. an upload's original name when ``path`` is a
        temp file. Returns the chunk ids.
        """
        source = source if source is not None else str(path)
        title = load_title(path) or Path(source).stem
        return self.ingest_pages(load_pages(path), source=source, metadata=metadata, title=title)

    def query(self, question: str, k: int = 5) -> QueryResult:
        """Retrieve relevant chunks and generate a grounded answer."""
        hits = self._retriever.retrieve(question, k=k)
        answer = generate_answer(question, hits, self._llm)
        return QueryResult(answer=answer, hits=hits)

    def retrieve(self, question: str, k: int = 5) -> list[Hit]:
        """Return the chunks retrieval surfaces for ``question`` -- no generation.

        For inspecting/tuning retrieval (scores, ranking) without an LLM call.
        """
        return self._retriever.retrieve(question, k=k)

    def list_chunks(self, limit: int = 100) -> list[tuple[str, dict[str, Any]]]:
        """Return up to ``limit`` indexed ``(id, metadata)`` pairs, for inspection."""
        return self._retriever.all_chunks(limit=limit)

    def delete_source(self, source: str) -> int:
        """Remove every chunk previously ingested from ``source``. Returns the count.

        Lets a scheduled re-ingest replace a changed file's chunks (delete the old
        set, then re-ingest) instead of leaving stale duplicates behind.
        """
        return self._retriever.delete_source(source)
