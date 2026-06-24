"""Integration tests for PgIngestStateStore against a real Postgres.

Run with a database available (e.g. `docker compose up -d`) via:

    pytest -m integration

These are skipped from the default unit run. They assert PgIngestStateStore
honors the same IngestStateStore contract as InMemoryIngestStateStore, and that
the config + manifest + last-run survive across store instances (i.e. restarts).
"""

import uuid
from collections.abc import Iterator

import psycopg
import pytest

from industryiq.config import get_settings
from industryiq.core.ingestion.adapters.store_pg import PgIngestStateStore
from industryiq.core.ingestion.models import FileState, IngestRunResult
from industryiq.core.ingestion.ports import IngestStateStore

pytestmark = pytest.mark.integration

DATABASE_URL = get_settings().database_url


@pytest.fixture
def tables() -> Iterator[tuple[str, str]]:
    if not DATABASE_URL:
        pytest.skip("DATABASE_URL not set")
    suffix = uuid.uuid4().hex[:8]
    job_table = f"test_ingest_job_{suffix}"
    files_table = f"test_ingest_files_{suffix}"
    yield job_table, files_table
    with psycopg.connect(DATABASE_URL) as conn:
        conn.execute(f"DROP TABLE IF EXISTS {job_table}")
        conn.execute(f"DROP TABLE IF EXISTS {files_table}")
        conn.commit()


def _store(tables: tuple[str, str]) -> PgIngestStateStore:
    assert DATABASE_URL is not None
    return PgIngestStateStore(DATABASE_URL, job_table=tables[0], files_table=tables[1])


def test_satisfies_interface(tables: tuple[str, str]) -> None:
    assert isinstance(_store(tables), IngestStateStore)


def test_default_config_is_disabled(tables: tuple[str, str]) -> None:
    config = _store(tables).get_config()
    assert config.enabled is False
    assert config.path is None
    assert config.interval_minutes == 60


def test_config_persists_across_instances(tables: tuple[str, str]) -> None:
    _store(tables).set_config(path="/data/reports", interval_minutes=15, enabled=True)
    # A fresh store instance == a restart; the row must already exist.
    reloaded = _store(tables).get_config()
    assert reloaded.path == "/data/reports"
    assert reloaded.interval_minutes == 15
    assert reloaded.enabled is True


def test_file_state_upsert_and_persist(tables: tuple[str, str]) -> None:
    store = _store(tables)
    assert store.get_file_state("a.txt") is None
    store.upsert_file_state(FileState(source="a.txt", size=10, content_hash="h1", chunk_count=2))
    store.upsert_file_state(FileState(source="a.txt", size=20, content_hash="h2", chunk_count=3))

    reloaded = _store(tables).get_file_state("a.txt")
    assert reloaded is not None
    assert reloaded.content_hash == "h2"
    assert reloaded.chunk_count == 3
    assert len(_store(tables).list_file_states()) == 1


def test_last_run_round_trips(tables: tuple[str, str]) -> None:
    from datetime import UTC, datetime

    store = _store(tables)
    assert store.last_run() is None
    result = IngestRunResult(
        ingested=2,
        updated=1,
        skipped=3,
        deleted_chunks=4,
        by_category={"AI": 5},
        failures=[("bad.pdf", "PdfReadError: boom")],
        started_at=datetime.now(UTC),
        finished_at=datetime.now(UTC),
    )
    store.record_run(result)

    reloaded = _store(tables).last_run()
    assert reloaded is not None
    assert reloaded.ingested == 2
    assert reloaded.by_category == {"AI": 5}
    assert reloaded.failures == [("bad.pdf", "PdfReadError: boom")]
