"""In-memory ingestion state store: the default (no DATABASE_URL) and test double.

Dict-backed, mirroring the other in-memory adapters. Satisfies
:class:`industryiq.core.ingestion.ports.IngestStateStore`, so the service and
scheduler cannot tell it apart from the Postgres-backed store. Because it does
not persist, the manifest is empty on each process start -- so without a database
every run re-ingests the whole tree (fine for local/dev; the live service uses
Postgres).
"""

from datetime import UTC, datetime

from industryiq.core.ingestion.models import FileState, IngestJobConfig, IngestRunResult
from industryiq.core.ingestion.ports import IngestStateStore


class InMemoryIngestStateStore(IngestStateStore):
    """A dict-backed ingestion state store for tests and local development."""

    def __init__(self) -> None:
        # Default: disabled, no path, hourly -- the admin turns it on via the API.
        self._config = IngestJobConfig(path=None, interval_minutes=60, enabled=False)
        self._files: dict[str, FileState] = {}
        self._last_run: IngestRunResult | None = None

    def get_config(self) -> IngestJobConfig:
        return self._config

    def set_config(
        self, *, path: str | None, interval_minutes: int, enabled: bool
    ) -> IngestJobConfig:
        self._config = IngestJobConfig(
            path=path,
            interval_minutes=interval_minutes,
            enabled=enabled,
            updated_at=datetime.now(UTC),
        )
        return self._config

    def get_file_state(self, source: str) -> FileState | None:
        return self._files.get(source)

    def upsert_file_state(self, state: FileState) -> None:
        self._files[state.source] = state

    def list_file_states(self) -> list[FileState]:
        return list(self._files.values())

    def record_run(self, result: IngestRunResult) -> None:
        self._last_run = result

    def last_run(self) -> IngestRunResult | None:
        return self._last_run
