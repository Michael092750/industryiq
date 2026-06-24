"""Scheduled bulk ingestion: scan a folder and load files into the shared KB.

Public surface, built as ports-and-adapters like :mod:`industryiq.core.auth` and
:mod:`industryiq.core.chat`:

* :class:`IngestionService` -- the idempotent scan-and-ingest policy.
* :class:`IngestScheduler` -- the background loop that runs it on an interval.
* :class:`IngestStateStore` (:mod:`.ports`) -- the persistence abstraction.
* :class:`InMemoryIngestStateStore` -- the default store and test double. The
  Postgres store is imported only where it is wired (:mod:`industryiq.api.deps`),
  to keep this package import light.
* The value types in :mod:`.models`.
"""

from industryiq.core.ingestion.adapters.store_memory import InMemoryIngestStateStore
from industryiq.core.ingestion.models import FileState, IngestJobConfig, IngestRunResult
from industryiq.core.ingestion.ports import IngestStateStore
from industryiq.core.ingestion.scheduler import IngestScheduler
from industryiq.core.ingestion.service import IngestionPathError, IngestionService

__all__ = [
    "FileState",
    "InMemoryIngestStateStore",
    "IngestJobConfig",
    "IngestRunResult",
    "IngestScheduler",
    "IngestStateStore",
    "IngestionPathError",
    "IngestionService",
]
