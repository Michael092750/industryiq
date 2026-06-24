from pathlib import Path

from industryiq.core.embeddings import FakeEmbedder
from industryiq.core.generation import FakeLLM
from industryiq.core.pipeline import RagPipeline
from industryiq.core.retrieval import Retriever
from industryiq.core.vectorstore import InMemoryVectorStore


def _pipeline(llm: FakeLLM | None = None, **kwargs: int) -> RagPipeline:
    retriever = Retriever(FakeEmbedder(dim=16), InMemoryVectorStore())
    return RagPipeline(retriever, llm or FakeLLM(), **kwargs)


def test_ingest_and_query_end_to_end() -> None:
    pipeline = _pipeline(FakeLLM(response="Cats are great [1]."))
    pipeline.ingest_text("cats are great and dogs are loyal")
    result = pipeline.query("cats are great and dogs are loyal")
    assert result.answer == "Cats are great [1]."
    assert result.hits
    assert "cats are great" in result.hits[0].metadata["text"]


def test_query_grounds_llm_prompt_in_retrieved_context() -> None:
    llm = FakeLLM()
    pipeline = _pipeline(llm, chunk_size=3, overlap=0)
    pipeline.ingest_text("alpha beta gamma delta epsilon zeta")
    pipeline.query("alpha beta gamma")
    assert llm.last_prompt is not None
    assert "alpha beta gamma" in llm.last_prompt


def test_ingest_text_with_source_records_metadata() -> None:
    pipeline = _pipeline(chunk_size=3, overlap=0)
    pipeline.ingest_text("alpha beta gamma", source="notes.txt")
    hit = pipeline.query("alpha beta gamma", k=1).hits[0]
    assert hit.metadata["source"] == "notes.txt"
    assert hit.metadata["text"] == "alpha beta gamma"


def test_ingest_text_with_metadata_records_extra_fields() -> None:
    pipeline = _pipeline(chunk_size=3, overlap=0)
    pipeline.ingest_text("alpha beta gamma", source="r.pdf", metadata={"category": "AI"})
    hit = pipeline.query("alpha beta gamma", k=1).hits[0]
    assert hit.metadata["category"] == "AI"
    assert hit.metadata["source"] == "r.pdf"


def test_ingest_file_reads_and_indexes(tmp_path: Path) -> None:
    file = tmp_path / "doc.txt"
    file.write_text("the quick brown fox jumps")
    pipeline = _pipeline()
    ids = pipeline.ingest_file(file)
    assert len(ids) == 1
    result = pipeline.query("the quick brown fox jumps", k=1)
    assert result.hits[0].metadata["source"] == str(file)


def test_ingest_file_with_metadata_records_category(tmp_path: Path) -> None:
    file = tmp_path / "doc.txt"
    file.write_text("the quick brown fox jumps")
    pipeline = _pipeline()
    pipeline.ingest_file(file, metadata={"category": "finance"})
    hit = pipeline.query("the quick brown fox jumps", k=1).hits[0]
    assert hit.metadata["category"] == "finance"
    assert hit.metadata["source"] == str(file)


def test_ingest_pages_tags_page_title_and_metadata() -> None:
    pipeline = _pipeline(chunk_size=3, overlap=0)
    ids = pipeline.ingest_pages(
        ["alpha beta gamma", "delta epsilon zeta"],
        source="report.pdf",
        metadata={"category": "AI"},
        title="My Report",
    )
    assert len(ids) == 2
    by_text = {meta["text"]: meta for _id, meta in pipeline.list_chunks()}
    assert by_text["alpha beta gamma"]["page"] == 1
    assert by_text["delta epsilon zeta"]["page"] == 2
    assert by_text["alpha beta gamma"]["title"] == "My Report"
    assert by_text["alpha beta gamma"]["category"] == "AI"
    assert by_text["alpha beta gamma"]["source"] == "report.pdf"


def test_ingest_pages_tags_section_from_markdown_headings() -> None:
    pipeline = _pipeline(chunk_size=50, overlap=0)
    pipeline.ingest_pages(
        ["# Report\nintro words\n## Outlook\nforecast words"],
        source="r.pdf",
        title="R",
    )
    by_section = {meta.get("section"): meta["text"] for _id, meta in pipeline.list_chunks()}
    assert by_section["Report"] == "# Report intro words"
    assert by_section["Outlook"] == "## Outlook forecast words"


def test_ingest_pages_carries_section_across_page_boundary() -> None:
    pipeline = _pipeline(chunk_size=50, overlap=0)
    # Heading opens on page 1; page 2 continues under it with no new heading.
    pipeline.ingest_pages(
        ["## Methodology\nstep one", "step two continues"],
        source="r.pdf",
        title="R",
    )
    by_page = {meta["page"]: meta for _id, meta in pipeline.list_chunks()}
    assert by_page[1]["section"] == "Methodology"
    assert by_page[2]["section"] == "Methodology"  # carried across the boundary


def test_ingest_pages_without_headings_leaves_section_unset() -> None:
    pipeline = _pipeline(chunk_size=10, overlap=0)
    pipeline.ingest_pages(["just plain words"], source="n.txt", title="N")
    meta = pipeline.list_chunks()[0][1]
    assert "section" not in meta


def test_ingest_pages_single_page_omits_page() -> None:
    pipeline = _pipeline()
    pipeline.ingest_pages(["just one page of text"], source="note.txt", title="Note")
    meta = pipeline.list_chunks()[0][1]
    assert "page" not in meta
    assert meta["title"] == "Note"


def test_ingest_file_sets_title_from_filename_stem(tmp_path: Path) -> None:
    file = tmp_path / "quarterly.txt"
    file.write_text("alpha beta gamma")
    pipeline = _pipeline()
    pipeline.ingest_file(file)
    meta = pipeline.list_chunks()[0][1]
    assert meta["title"] == "quarterly"  # no embedded title -> filename stem
    assert "page" not in meta  # txt is single-page


def test_list_chunks_returns_ingested_content() -> None:
    pipeline = _pipeline(chunk_size=3, overlap=0)
    pipeline.ingest_text("alpha beta gamma delta", source="doc.txt")
    items = pipeline.list_chunks()
    metadatas = [meta for _id, meta in items]
    assert all(meta["source"] == "doc.txt" for meta in metadatas)
    assert {meta["text"] for meta in metadatas} == {"alpha beta gamma", "delta"}


def test_query_on_empty_store_still_returns_answer() -> None:
    pipeline = _pipeline(FakeLLM(response="I don't know."))
    result = pipeline.query("anything")
    assert result.answer == "I don't know."
    assert result.hits == []


def test_retrieve_returns_hits_without_generating() -> None:
    llm = FakeLLM()
    pipeline = _pipeline(llm)
    pipeline.ingest_text("alpha beta gamma")
    hits = pipeline.retrieve("alpha beta gamma", k=1)
    assert "alpha beta gamma" in hits[0].metadata["text"]
    assert llm.last_prompt is None  # retrieval only -- no generation happened
