"""Value types for the scheduled bulk-ingestion job.

Three frozen dataclasses describe the job's state:

* :class:`IngestJobConfig` -- the admin-set policy (where to scan, how often,
  whether it's on). One per deployment, persisted so it survives a restart.
* :class:`FileState` -- the per-file manifest entry that makes re-runs idempotent:
  the content hash of the last-ingested version of a file, keyed by its ``source``
  (path relative to the scan root). A scan compares each file's current hash to
  this to decide skip / re-ingest.
* :class:`IngestRunResult` -- the outcome of one scan, for the admin readout.
"""

from dataclasses import dataclass, field
from datetime import datetime


@dataclass(frozen=True)
class IngestJobConfig:
    """The admin-configured ingestion policy (a single per-deployment record)."""

    path: str | None
    interval_minutes: int
    enabled: bool
    updated_at: datetime | None = None


@dataclass(frozen=True)
class FileState:
    """The last-ingested fingerprint of one file, keyed by ``source``.

    ``source`` is the file's path relative to the scan root (posix-normalized) --
    the same string stamped on every chunk's metadata, so it also keys the
    delete-by-source used to replace a changed file's chunks.
    """

    source: str
    size: int
    content_hash: str
    chunk_count: int
    ingested_at: datetime | None = None


@dataclass(frozen=True)
class IngestRunResult:
    """Counts and failures from a single ``run_once`` scan."""

    ingested: int = 0
    updated: int = 0
    skipped: int = 0
    deleted_chunks: int = 0
    by_category: dict[str, int] = field(default_factory=dict)
    failures: list[tuple[str, str]] = field(default_factory=list)
    started_at: datetime | None = None
    finished_at: datetime | None = None
    busy: bool = False

    @property
    def chunks_added(self) -> int:
        """Total chunks written across newly-ingested and updated files."""
        return sum(self.by_category.values())
