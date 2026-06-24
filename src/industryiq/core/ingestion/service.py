"""IngestionService: scan a folder and ingest new/changed files idempotently.

One ``run_once`` call walks the configured directory tree (each top-level
subfolder is a document *category*, as in ``scripts/ingest_bulk.py``) and, for
each supported file, decides by content hash whether to:

* **ingest** it (no manifest entry yet),
* **replace** it (hash changed -- delete the file's old chunks, then re-ingest),
* **skip** it (hash unchanged).

The content hash, not mtime, is the source of truth -- mtime is unreliable across
OneDrive sync and container bind-mounts. The file's ``source`` (path relative to
the scan root, posix-normalized) is the stable key shared by the manifest, the
chunk metadata, and the delete-by-source, so a changed file's *old* chunks (and
only those) are removed before its new ones go in.

The service is provider-agnostic: it drives a :class:`RagPipeline`, so the same
embedder/store the live app queries with is the one it writes to (no dim
mismatch). A lock makes ``run_once`` non-overlapping, so the background scheduler
and a manual "run now" can't collide.
"""

import hashlib
import threading
from collections import Counter
from collections.abc import Collection
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from industryiq.core.ingestion.manifest import ManifestCache, manifest_metadata
from industryiq.core.ingestion.models import FileState, IngestJobConfig, IngestRunResult
from industryiq.core.ingestion.ports import IngestStateStore
from industryiq.core.loaders import SUPPORTED_EXTENSIONS
from industryiq.core.pipeline import RagPipeline

# Read files in 1 MiB blocks when hashing, so a large PDF doesn't load whole.
_HASH_BLOCK = 1 << 20


class IngestionPathError(Exception):
    """The configured ingestion path is missing, empty, or not a directory."""


def _file_hash(path: Path) -> str:
    """sha256 hex digest of a file's bytes, read in blocks."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(_HASH_BLOCK), b""):
            digest.update(block)
    return digest.hexdigest()


class IngestionService:
    """Run an idempotent bulk-ingest of a directory tree through a pipeline."""

    def __init__(
        self,
        pipeline: RagPipeline,
        store: IngestStateStore,
        *,
        supported: Collection[str] = SUPPORTED_EXTENSIONS,
    ) -> None:
        self._pipeline = pipeline
        self._store = store
        self._supported = frozenset(s.lower() for s in supported)
        self._lock = threading.Lock()

    def config(self) -> IngestJobConfig:
        """The current job config (passthrough, for the scheduler/API)."""
        return self._store.get_config()

    def set_config(
        self, *, path: str | None, interval_minutes: int, enabled: bool
    ) -> IngestJobConfig:
        """Replace the job config (passthrough, for the admin API)."""
        return self._store.set_config(path=path, interval_minutes=interval_minutes, enabled=enabled)

    def last_run(self) -> IngestRunResult | None:
        """The most recent run's result, or ``None`` (passthrough, for the API)."""
        return self._store.last_run()

    def run_once(self, path: str | Path | None = None) -> IngestRunResult:
        """Scan ``path`` (or the configured path) once and ingest new/changed files.

        Returns a summary. If a run is already in progress the call is a no-op and
        returns a result flagged ``busy`` (the in-flight run is left untouched).
        """
        if not self._lock.acquire(blocking=False):
            now = datetime.now(UTC)
            return IngestRunResult(busy=True, started_at=now, finished_at=now)
        try:
            return self._run(path)
        finally:
            self._lock.release()

    def _run(self, path: str | Path | None) -> IngestRunResult:
        started = datetime.now(UTC)
        root_value = path if path is not None else self._store.get_config().path
        if not root_value:
            raise IngestionPathError("no ingestion path configured")
        root = Path(root_value)
        if not root.is_dir():
            raise IngestionPathError(f"not a directory: {root}")

        ingested = updated = skipped = deleted = 0
        by_category: Counter[str] = Counter()
        failures: list[tuple[str, str]] = []
        # Memoizes each category's manifest.csv so it is parsed once per scan.
        manifests: ManifestCache = {}

        for file_path in sorted(root.rglob("*")):
            if not file_path.is_file() or file_path.suffix.lower() not in self._supported:
                continue
            rel = file_path.relative_to(root)
            source = rel.as_posix()
            # Top-level subfolder = category; files directly under root are uncategorized.
            category = rel.parts[0] if len(rel.parts) > 1 else "uncategorized"
            try:
                digest = _file_hash(file_path)
                prior = self._store.get_file_state(source)
                if prior is not None and prior.content_hash == digest:
                    skipped += 1
                    continue
                if prior is not None:
                    # Replace: drop the old version's chunks before re-ingesting.
                    deleted += self._pipeline.delete_source(source)
                # Provenance (publisher/published_date/source_type) from the sibling
                # manifest, if any; category always wins as the canonical label.
                metadata: dict[str, Any] = {
                    **manifest_metadata(file_path, manifests),
                    "category": category,
                }
                ids = self._pipeline.ingest_file(file_path, source=source, metadata=metadata)
                if prior is None:
                    ingested += 1
                else:
                    updated += 1
                by_category[category] += len(ids)
                self._store.upsert_file_state(
                    FileState(
                        source=source,
                        size=file_path.stat().st_size,
                        content_hash=digest,
                        chunk_count=len(ids),
                        ingested_at=datetime.now(UTC),
                    )
                )
            except Exception as exc:  # noqa: BLE001 -- one bad file mustn't abort the batch
                failures.append((source, f"{type(exc).__name__}: {exc}"))

        result = IngestRunResult(
            ingested=ingested,
            updated=updated,
            skipped=skipped,
            deleted_chunks=deleted,
            by_category=dict(by_category),
            failures=failures,
            started_at=started,
            finished_at=datetime.now(UTC),
        )
        self._store.record_run(result)
        return result
