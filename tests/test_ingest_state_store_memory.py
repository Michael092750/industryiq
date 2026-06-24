"""Unit tests for the in-memory ingestion state store (the default / test double)."""

from datetime import UTC, datetime

from industryiq.core.ingestion import IngestStateStore, InMemoryIngestStateStore
from industryiq.core.ingestion.models import FileState, IngestRunResult


def test_satisfies_interface() -> None:
    assert isinstance(InMemoryIngestStateStore(), IngestStateStore)


def test_default_config_is_disabled() -> None:
    config = InMemoryIngestStateStore().get_config()
    assert config.enabled is False
    assert config.path is None
    assert config.interval_minutes == 60


def test_set_config_round_trips() -> None:
    store = InMemoryIngestStateStore()
    saved = store.set_config(path="/data/reports", interval_minutes=15, enabled=True)
    assert saved.path == "/data/reports"
    assert saved.interval_minutes == 15
    assert saved.enabled is True
    assert saved.updated_at is not None
    assert store.get_config() == saved


def test_file_state_upsert_get_and_list() -> None:
    store = InMemoryIngestStateStore()
    assert store.get_file_state("a.txt") is None

    state = FileState(
        source="a.txt",
        size=10,
        content_hash="abc",
        chunk_count=2,
        ingested_at=datetime.now(UTC),
    )
    store.upsert_file_state(state)
    assert store.get_file_state("a.txt") == state

    # Upsert replaces in place (no duplicate entry).
    store.upsert_file_state(FileState(source="a.txt", size=20, content_hash="def", chunk_count=3))
    assert store.get_file_state("a.txt").content_hash == "def"
    assert len(store.list_file_states()) == 1


def test_record_and_read_last_run() -> None:
    store = InMemoryIngestStateStore()
    assert store.last_run() is None
    result = IngestRunResult(ingested=3, skipped=1)
    store.record_run(result)
    assert store.last_run() == result
