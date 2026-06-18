"""Bulk-ingest a folder tree of reports into the shared knowledge base.

Each top-level subfolder is treated as a document *category* (industry):

    reports/AI/1.pdf       -> category "AI"
    reports/finance/2.pdf  -> category "finance"

The category is stored in each chunk's metadata, so retrieved hits are
attributable to an industry. Subfolders are walked recursively; only the first
path segment under the root is used as the category.

Writes directly through RagPipeline (no HTTP), so it needs a *shared* store:
set DATABASE_URL to your Postgres so the running API sees the same data. Run the
SAME RAG_PROVIDER as the server, so the ingest-time and query-time embedders
match (a 384-dim local embedder and 1024-dim Titan are not interchangeable).

Usage:
    python scripts/ingest_bulk.py <folder>
"""

import sys
from collections import Counter
from pathlib import Path

from ragproject.api.deps import get_pipeline
from ragproject.core.loaders import SUPPORTED_EXTENSIONS, load


def main(root: Path) -> None:
    pipeline = get_pipeline()
    counts: Counter[str] = Counter()
    failures: list[tuple[Path, str]] = []
    for path in sorted(root.rglob("*")):
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        rel = path.relative_to(root)
        # Top-level subfolder = category; files directly under root are uncategorized.
        category = rel.parts[0] if len(rel.parts) > 1 else "uncategorized"
        try:
            ids = pipeline.ingest_text(load(path), source=str(rel), metadata={"category": category})
        except Exception as exc:
            # One unreadable file (encrypted, corrupt, ...) shouldn't abort the batch.
            failures.append((rel, f"{type(exc).__name__}: {exc}"))
            print(f"[skip] {rel} -> {type(exc).__name__}: {exc}")
            continue
        counts[category] += len(ids)
        print(f"[{category}] {rel} -> {len(ids)} chunks")

    total = sum(counts.values())
    print(f"\nDone: {total} chunks across {len(counts)} categories")
    for category, n in sorted(counts.items()):
        print(f"  {category}: {n}")
    if failures:
        print(f"\nSkipped {len(failures)} file(s):")
        for rel, msg in failures:
            print(f"  {rel}: {msg}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: python scripts/ingest_bulk.py <folder>")
    folder = Path(sys.argv[1])
    if not folder.is_dir():
        sys.exit(f"not a folder: {folder}")
    main(folder)
