"""Unit tests for IngestionService: idempotent, manifest-driven directory scans.

All offline: a real :class:`RagPipeline` over fakes + in-memory stores, scanning
files written under ``tmp_path``. These pin the skip / re-ingest / replace logic
that makes a scheduled re-scan safe.
"""

from pathlib import Path

import pytest

from industryiq.core.embeddings import FakeEmbedder
from industryiq.core.generation import FakeLLM
from industryiq.core.ingestion import (
    IngestionPathError,
    IngestionService,
    InMemoryIngestStateStore,
)
from industryiq.core.pipeline import RagPipeline
from industryiq.core.retrieval import Retriever
from industryiq.core.vectorstore import InMemoryVectorStore


@pytest.fixture
def store() -> InMemoryVectorStore:
    return InMemoryVectorStore()


@pytest.fixture
def service(store: InMemoryVectorStore) -> IngestionService:
    pipeline = RagPipeline(Retriever(FakeEmbedder(dim=16), store), FakeLLM())
    return IngestionService(pipeline, InMemoryIngestStateStore())


def _write(root: Path, rel: str, text: str) -> Path:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def test_ingests_new_files_with_category_from_subfolder(
    service: IngestionService, store: InMemoryVectorStore, tmp_path: Path
) -> None:
    _write(tmp_path, "AI/forecast.txt", "alpha beta gamma")
    _write(tmp_path, "finance/risk.txt", "delta epsilon")
    _write(tmp_path, "loose.txt", "top-level file")

    result = service.run_once(tmp_path)

    assert result.ingested == 3
    assert result.updated == 0
    assert result.skipped == 0
    assert set(result.by_category) == {"AI", "finance", "uncategorized"}
    categories = {meta["category"] for _, meta in store.all_items()}
    assert categories == {"AI", "finance", "uncategorized"}
    # source is the posix relative path -- stable, forward slashes.
    sources = {meta["source"] for _, meta in store.all_items()}
    assert "AI/forecast.txt" in sources


def test_manifest_provenance_is_attached_to_chunks(
    service: IngestionService, store: InMemoryVectorStore, tmp_path: Path
) -> None:
    _write(tmp_path, "AI/report.txt", "alpha beta gamma")
    # A sibling manifest.csv supplies provenance, joined by filename. It is itself
    # an unsupported extension, so the scan ignores it as a document.
    _write(
        tmp_path,
        "AI/manifest.csv",
        "status,url,filename,domain,detected_year,size_bytes,sha256,error\n"
        "downloaded,https://www.mckinsey.com/r.pdf,report.txt,www.mckinsey.com,2024,1,abc,\n",
    )

    service.run_once(tmp_path)

    meta = next(m for _id, m in store.all_items() if m["source"] == "AI/report.txt")
    assert meta["publisher"] == "mckinsey.com"
    assert meta["source_type"] == "consultancy"
    assert meta["published_date"] == "2024"
    assert meta["category"] == "AI"  # category still the canonical label


def test_files_without_a_manifest_get_only_category(
    service: IngestionService, store: InMemoryVectorStore, tmp_path: Path
) -> None:
    _write(tmp_path, "AI/a.txt", "alpha beta gamma")

    service.run_once(tmp_path)

    meta = store.all_items()[0][1]
    assert meta["category"] == "AI"
    assert "publisher" not in meta
    assert "source_type" not in meta
    assert "published_date" not in meta


def test_unchanged_files_are_skipped_on_rerun(
    service: IngestionService, store: InMemoryVectorStore, tmp_path: Path
) -> None:
    _write(tmp_path, "AI/a.txt", "stable content")
    first = service.run_once(tmp_path)
    before = dict(store.all_items())

    second = service.run_once(tmp_path)

    assert first.ingested == 1
    assert second.ingested == 0
    assert second.updated == 0
    assert second.skipped == 1
    # No churn: the stored chunks are byte-for-byte the same ids/metadata.
    assert dict(store.all_items()) == before


def test_changed_file_is_replaced_not_duplicated(
    service: IngestionService, store: InMemoryVectorStore, tmp_path: Path
) -> None:
    path = _write(tmp_path, "AI/a.txt", "original text")
    service.run_once(tmp_path)

    path.write_text("completely different text now", encoding="utf-8")
    result = service.run_once(tmp_path)

    assert result.ingested == 0
    assert result.updated == 1
    assert result.deleted_chunks >= 1
    texts = {meta["text"] for _, meta in store.all_items()}
    assert texts == {"completely different text now"}  # old chunk is gone


def test_unsupported_files_are_ignored(
    service: IngestionService, store: InMemoryVectorStore, tmp_path: Path
) -> None:
    _write(tmp_path, "AI/a.txt", "keep me")
    _write(tmp_path, "AI/notes.csv", "a,b,c")
    _write(tmp_path, "AI/image.png", "not really an image")

    result = service.run_once(tmp_path)

    assert result.ingested == 1
    assert all(meta["source"] == "AI/a.txt" for _, meta in store.all_items())


def test_one_bad_file_is_recorded_but_batch_continues(
    service: IngestionService, store: InMemoryVectorStore, tmp_path: Path
) -> None:
    _write(tmp_path, "AI/good.txt", "fine content")
    # A 0-byte .pdf makes the PDF loader raise -- a realistic per-file failure.
    (tmp_path / "AI" / "broken.pdf").write_bytes(b"")

    result = service.run_once(tmp_path)

    assert result.ingested == 1  # the good file still went in
    assert len(result.failures) == 1
    failed_source, message = result.failures[0]
    assert failed_source == "AI/broken.pdf"
    assert message  # carries the exception type/message


def test_run_records_last_run_summary(service: IngestionService, tmp_path: Path) -> None:
    _write(tmp_path, "AI/a.txt", "content")
    service.run_once(tmp_path)
    last = service.last_run()
    assert last is not None
    assert last.ingested == 1
    assert last.finished_at is not None


def test_run_without_path_or_config_raises(service: IngestionService) -> None:
    with pytest.raises(IngestionPathError):
        service.run_once()


def test_run_uses_configured_path_when_arg_omitted(
    service: IngestionService, tmp_path: Path
) -> None:
    _write(tmp_path, "AI/a.txt", "content")
    service.set_config(path=str(tmp_path), interval_minutes=30, enabled=True)
    result = service.run_once()
    assert result.ingested == 1


def test_missing_directory_raises(service: IngestionService, tmp_path: Path) -> None:
    with pytest.raises(IngestionPathError):
        service.run_once(tmp_path / "does-not-exist")


def test_concurrent_run_returns_busy(service: IngestionService, tmp_path: Path) -> None:
    _write(tmp_path, "AI/a.txt", "content")
    # Hold the run lock so the next call's non-blocking acquire fails -> busy.
    assert service._lock.acquire(blocking=False)
    try:
        result = service.run_once(tmp_path)
    finally:
        service._lock.release()
    assert result.busy is True
    assert result.ingested == 0
