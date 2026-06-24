"""Bulk-ingest a folder tree of reports into the shared knowledge base.

Each top-level subfolder is treated as a document *category* (industry):

    reports/AI/1.pdf       -> category "AI"
    reports/finance/2.pdf  -> category "finance"

The category is stored in each chunk's metadata, so retrieved hits are
attributable to an industry. Subfolders are walked recursively; only the first
path segment under the root is used as the category.

This is a thin CLI over :class:`industryiq.core.ingestion.IngestionService` --
the *same* code the scheduled job runs in the live service. It writes through the
configured pipeline (Milvus when ``VECTOR_BACKEND=milvus``, else pgvector/
in-memory) and is idempotent: a re-run skips unchanged files and replaces changed
ones, tracked in the manifest. The manifest is durable only with ``DATABASE_URL``
set; without it the run still works but each run re-ingests the whole tree.

Run with the SAME ``RAG_PROVIDER`` as the server, so the ingest-time and
query-time embedders match (a 384-dim local embedder and 1024-dim Titan are not
interchangeable against one store).

Usage:
    python scripts/ingest_bulk.py <folder>
"""

import sys
from pathlib import Path

from industryiq.api.deps import get_ingestion_service


def main(root: Path) -> None:
    result = get_ingestion_service().run_once(root)

    print(
        f"\nDone: {result.ingested} new, {result.updated} updated, "
        f"{result.skipped} unchanged; {result.chunks_added} chunks across "
        f"{len(result.by_category)} categories"
    )
    for category, n in sorted(result.by_category.items()):
        print(f"  {category}: {n}")
    if result.deleted_chunks:
        print(f"replaced (deleted) {result.deleted_chunks} stale chunk(s)")
    if result.failures:
        print(f"\nSkipped {len(result.failures)} file(s):")
        for source, msg in result.failures:
            print(f"  {source}: {msg}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: python scripts/ingest_bulk.py <folder>")
    folder = Path(sys.argv[1])
    if not folder.is_dir():
        sys.exit(f"not a folder: {folder}")
    main(folder)
