"""Lossless in-place reindex of the Milvus chunk collection to promote chunk_index.

The old collection stored chunk_index inside the residual ``metadata`` JSON blob; the
new schema promotes it to an indexed INT64 column (needed by fetch_neighbors /
context expansion). This copies every row -- SAME ids, SAME embeddings (read back,
never re-embedded), SAME metadata values -- into a freshly-created collection with the
new schema, then swaps it in. The chunk_index value is the one already stored per row;
it is never recomputed.

Safety: builds into a temp collection and only drops/renames the original AFTER the
new one is verified (matching row count + populated chunk_index). The source is
untouched until then, so a crash mid-run loses nothing (drop the temp and re-run). No
pgvector, no PDF re-parse, no re-embed.

Run with the SAME env as the server (VECTOR_BACKEND=milvus, MILVUS_URI, ...):

    python scripts/reindex_milvus_chunk_index.py             # build + verify + swap
    python scripts/reindex_milvus_chunk_index.py --dry-run   # build + verify, do NOT swap
    python scripts/reindex_milvus_chunk_index.py --swap-only  # verify an existing temp + swap
"""

import argparse
import sys

from pymilvus import MilvusClient

from industryiq.config import get_settings
from industryiq.core.milvusvectorstore import MilvusVectorStore

# Old-schema fields to read (the old collection has NO chunk_index column -- that
# value lives in the `metadata` JSON blob, which _merge_metadata folds back in).
_READ_FIELDS = [
    "id",
    "embedding",
    "text",
    "source",
    "section",
    "category",
    "publisher",
    "source_type",
    "published_date",
    "metadata",
]
_BATCH = 1000


def _row_count(client: MilvusClient, name: str) -> int:
    client.flush(name)
    return int(client.get_collection_stats(name)["row_count"])


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="build + verify, skip the swap")
    parser.add_argument(
        "--swap-only",
        action="store_true",
        help="skip building; verify an existing temp from a prior --dry-run and swap it in",
    )
    args = parser.parse_args(argv)

    s = get_settings()
    src = s.milvus_collection
    tmp = f"{src}_reindex"
    client = MilvusClient(uri=s.milvus_uri, token=s.milvus_token or "")

    if not client.has_collection(src):
        sys.exit(f"source collection {src!r} does not exist")
    client.load_collection(src)
    total = _row_count(client, src)
    print(f"source {src!r}: {total} rows")

    if args.swap_only:
        if not client.has_collection(tmp):
            sys.exit(f"--swap-only: temp {tmp!r} does not exist; run a --dry-run first")
        client.load_collection(tmp)
        dest_count = _row_count(client, tmp)
        print(f"temp {tmp!r}: {dest_count} rows (source {total})")
        if dest_count != total:
            sys.exit(f"ABORT: count mismatch ({dest_count} != {total}); source left intact.")
        probe = client.query(tmp, filter="", output_fields=["id", "chunk_index"], limit=5)
        print(f"spot-check chunk_index values: {[r.get('chunk_index') for r in probe]}")
        if not probe or all(int(r.get("chunk_index", -1)) < 0 for r in probe):
            sys.exit(f"ABORT: chunk_index not populated in {tmp!r}; source left intact.")
        client.drop_collection(src)
        client.rename_collection(tmp, src)
        print(f"\nSwapped {tmp!r} -> {src!r}. chunk_index is now a promoted, indexed column.")
        return 0

    dim = len(client.query(src, filter="", output_fields=["embedding"], limit=1)[0]["embedding"])
    print(f"embedding dim = {dim}")

    # Fresh temp collection with the NEW schema (promoted+indexed chunk_index).
    if client.has_collection(tmp):
        client.drop_collection(tmp)
        print(f"dropped stale temp {tmp!r}")
    dest = MilvusVectorStore(
        s.milvus_uri,
        dim=dim,
        collection=tmp,
        token=s.milvus_token,
        index_type=s.milvus_index_type,
    )

    it = client.query_iterator(
        collection_name=src, filter="", output_fields=_READ_FIELDS, batch_size=_BATCH
    )
    moved = 0
    try:
        while True:
            batch = it.next()
            if not batch:
                break
            ids = [row["id"] for row in batch]
            vectors = [row["embedding"] for row in batch]
            # _merge_metadata reconstructs the full original dict (incl chunk_index
            # from the JSON blob); dest.upsert's _split_metadata then promotes it.
            metas = [MilvusVectorStore._merge_metadata(row) for row in batch]
            dest.upsert(ids, vectors, metas)
            moved += len(ids)
            print(f"  reindexed {moved}/{total}", flush=True)
    finally:
        it.close()

    dest_count = _row_count(client, tmp)
    print(f"\ndest {tmp!r}: {dest_count} rows (source had {total})")
    if dest_count != total:
        sys.exit(
            f"ABORT: count mismatch ({dest_count} != {total}). Source {src!r} left "
            f"intact; temp {tmp!r} kept for inspection."
        )

    # Spot-check: chunk_index is now a real promoted column, populated (not the -1
    # "absent" sentinel) on the sampled rows.
    probe = client.query(tmp, filter="", output_fields=["id", "chunk_index"], limit=5)
    print(f"spot-check dest chunk_index values: {[r.get('chunk_index') for r in probe]}")
    if not probe or all(int(r.get("chunk_index", -1)) < 0 for r in probe):
        sys.exit(f"ABORT: chunk_index not populated in {tmp!r}; source left intact.")

    if args.dry_run:
        print(f"\n--dry-run: NOT swapping. Verified temp {tmp!r} has {dest_count} rows.")
        return 0

    client.drop_collection(src)
    client.rename_collection(tmp, src)
    print(f"\nSwapped {tmp!r} -> {src!r}. chunk_index is now a promoted, indexed column.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
