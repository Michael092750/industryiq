"""The ingestion module's port -- the persistence abstraction it depends on.

Following the same ports-and-adapters style as :mod:`industryiq.core.auth` and
:mod:`industryiq.core.chat`, :class:`IngestionService` and the scheduler are
written against this ``Protocol``, never a concrete store. An in-memory adapter
backs tests and the no-database path; a Postgres adapter backs the live service.

The store holds three things: the singleton job config, the per-file manifest
(for idempotent re-runs), and the last run's result (for the admin readout).
"""

from typing import Protocol, runtime_checkable

from industryiq.core.ingestion.models import FileState, IngestJobConfig, IngestRunResult


@runtime_checkable
class IngestStateStore(Protocol):
    """Persist the ingestion job's config, file manifest, and last-run summary."""

    def get_config(self) -> IngestJobConfig:
        """Return the current config (a disabled default when never set)."""
        ...

    def set_config(
        self, *, path: str | None, interval_minutes: int, enabled: bool
    ) -> IngestJobConfig:
        """Replace the job config and return the stored record."""
        ...

    def get_file_state(self, source: str) -> FileState | None:
        """Return the manifest entry for ``source``, or ``None`` if never ingested."""
        ...

    def upsert_file_state(self, state: FileState) -> None:
        """Insert or replace the manifest entry for ``state.source``."""
        ...

    def list_file_states(self) -> list[FileState]:
        """Return every manifest entry (for inspection / status)."""
        ...

    def record_run(self, result: IngestRunResult) -> None:
        """Store ``result`` as the most recent run."""
        ...

    def last_run(self) -> IngestRunResult | None:
        """Return the most recent run's result, or ``None`` if none has run."""
        ...
