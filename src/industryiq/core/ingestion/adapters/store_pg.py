"""A Postgres implementation of the IngestStateStore interface.

Follows the conventions of :class:`industryiq.core.chat.adapters.store_pg`: raw
``psycopg``, tables created on construction with ``CREATE TABLE IF NOT EXISTS``,
and one connection per operation for thread-safety under the web server.

Two tables back the three concerns of the port:

* ``ingest_job`` -- a single row (``id = 'default'``) holding the admin config
  plus the last run's summary (as JSONB). The row is seeded on construction so
  config reads and run records always have something to update.
* ``ingest_files`` -- the per-file manifest (``source`` -> content hash), which
  makes scheduled re-runs idempotent.
"""

from datetime import UTC, datetime
from typing import Any

import psycopg
from psycopg.types.json import Jsonb

from industryiq.core.ingestion.models import FileState, IngestJobConfig, IngestRunResult
from industryiq.core.ingestion.ports import IngestStateStore

_CONFIG_ID = "default"


def _parse_dt(value: Any) -> datetime | None:
    return datetime.fromisoformat(value) if value else None


def _result_to_json(result: IngestRunResult) -> dict[str, Any]:
    return {
        "ingested": result.ingested,
        "updated": result.updated,
        "skipped": result.skipped,
        "deleted_chunks": result.deleted_chunks,
        "by_category": result.by_category,
        "failures": [list(failure) for failure in result.failures],
        "started_at": result.started_at.isoformat() if result.started_at else None,
        "finished_at": result.finished_at.isoformat() if result.finished_at else None,
        "busy": result.busy,
    }


def _json_to_result(data: dict[str, Any] | None) -> IngestRunResult | None:
    if not data:
        return None
    return IngestRunResult(
        ingested=data.get("ingested", 0),
        updated=data.get("updated", 0),
        skipped=data.get("skipped", 0),
        deleted_chunks=data.get("deleted_chunks", 0),
        by_category=dict(data.get("by_category") or {}),
        failures=[tuple(failure) for failure in data.get("failures") or []],
        started_at=_parse_dt(data.get("started_at")),
        finished_at=_parse_dt(data.get("finished_at")),
        busy=data.get("busy", False),
    )


class PgIngestStateStore(IngestStateStore):
    """Ingestion state store backed by two Postgres tables."""

    def __init__(
        self,
        dsn: str,
        *,
        job_table: str = "ingest_job",
        files_table: str = "ingest_files",
    ) -> None:
        self._dsn = dsn
        self._job_table = job_table
        self._files_table = files_table
        with psycopg.connect(dsn) as conn:
            conn.execute(
                f"CREATE TABLE IF NOT EXISTS {job_table} ("
                f"id TEXT PRIMARY KEY, path TEXT, "
                f"interval_minutes INT NOT NULL DEFAULT 60, "
                f"enabled BOOLEAN NOT NULL DEFAULT false, "
                f"last_run_at TIMESTAMPTZ, last_run_summary JSONB, "
                f"updated_at TIMESTAMPTZ)"
            )
            # Seed the singleton row so config reads and run records always update.
            conn.execute(
                f"INSERT INTO {job_table} (id) VALUES (%s) ON CONFLICT (id) DO NOTHING",
                (_CONFIG_ID,),
            )
            conn.execute(
                f"CREATE TABLE IF NOT EXISTS {files_table} ("
                f"source TEXT PRIMARY KEY, size BIGINT NOT NULL, "
                f"content_hash TEXT NOT NULL, chunk_count INT NOT NULL, "
                f"ingested_at TIMESTAMPTZ NOT NULL DEFAULT now())"
            )
            conn.commit()

    def get_config(self) -> IngestJobConfig:
        with psycopg.connect(self._dsn) as conn:
            row = conn.execute(
                f"SELECT path, interval_minutes, enabled, updated_at "
                f"FROM {self._job_table} WHERE id = %s",
                (_CONFIG_ID,),
            ).fetchone()
        if row is None:  # only if the seed row was deleted out from under us
            return IngestJobConfig(path=None, interval_minutes=60, enabled=False)
        return IngestJobConfig(
            path=row[0], interval_minutes=row[1], enabled=row[2], updated_at=row[3]
        )

    def set_config(
        self, *, path: str | None, interval_minutes: int, enabled: bool
    ) -> IngestJobConfig:
        updated_at = datetime.now(UTC)
        with psycopg.connect(self._dsn) as conn:
            conn.execute(
                f"INSERT INTO {self._job_table} "
                f"(id, path, interval_minutes, enabled, updated_at) "
                f"VALUES (%s, %s, %s, %s, %s) "
                f"ON CONFLICT (id) DO UPDATE SET "
                f"path = EXCLUDED.path, interval_minutes = EXCLUDED.interval_minutes, "
                f"enabled = EXCLUDED.enabled, updated_at = EXCLUDED.updated_at",
                (_CONFIG_ID, path, interval_minutes, enabled, updated_at),
            )
            conn.commit()
        return IngestJobConfig(
            path=path, interval_minutes=interval_minutes, enabled=enabled, updated_at=updated_at
        )

    def get_file_state(self, source: str) -> FileState | None:
        with psycopg.connect(self._dsn) as conn:
            row = conn.execute(
                f"SELECT source, size, content_hash, chunk_count, ingested_at "
                f"FROM {self._files_table} WHERE source = %s",
                (source,),
            ).fetchone()
        if row is None:
            return None
        return FileState(
            source=row[0], size=row[1], content_hash=row[2], chunk_count=row[3], ingested_at=row[4]
        )

    def upsert_file_state(self, state: FileState) -> None:
        with psycopg.connect(self._dsn) as conn:
            conn.execute(
                f"INSERT INTO {self._files_table} "
                f"(source, size, content_hash, chunk_count, ingested_at) "
                f"VALUES (%s, %s, %s, %s, %s) "
                f"ON CONFLICT (source) DO UPDATE SET "
                f"size = EXCLUDED.size, content_hash = EXCLUDED.content_hash, "
                f"chunk_count = EXCLUDED.chunk_count, ingested_at = EXCLUDED.ingested_at",
                (
                    state.source,
                    state.size,
                    state.content_hash,
                    state.chunk_count,
                    state.ingested_at or datetime.now(UTC),
                ),
            )
            conn.commit()

    def list_file_states(self) -> list[FileState]:
        with psycopg.connect(self._dsn) as conn:
            rows = conn.execute(
                f"SELECT source, size, content_hash, chunk_count, ingested_at "
                f"FROM {self._files_table} ORDER BY source"
            ).fetchall()
        return [
            FileState(
                source=row[0],
                size=row[1],
                content_hash=row[2],
                chunk_count=row[3],
                ingested_at=row[4],
            )
            for row in rows
        ]

    def record_run(self, result: IngestRunResult) -> None:
        with psycopg.connect(self._dsn) as conn:
            conn.execute(
                f"UPDATE {self._job_table} SET last_run_at = %s, last_run_summary = %s "
                f"WHERE id = %s",
                (result.finished_at, Jsonb(_result_to_json(result)), _CONFIG_ID),
            )
            conn.commit()

    def last_run(self) -> IngestRunResult | None:
        with psycopg.connect(self._dsn) as conn:
            row = conn.execute(
                f"SELECT last_run_summary FROM {self._job_table} WHERE id = %s",
                (_CONFIG_ID,),
            ).fetchone()
        return _json_to_result(row[0] if row else None)
