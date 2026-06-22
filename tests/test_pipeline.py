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
