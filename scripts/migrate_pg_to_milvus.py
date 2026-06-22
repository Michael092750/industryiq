"""Copy the chunk corpus from Postgres+pgvector into Milvus.

Reads every chunk from the live pgvector ``chunks`` table and writes it into a
Milvus collection, **reusing the same string ids and metadata**. Vectors are not
copied out of Postgres; instead each chunk's stored text (``metadata["text"]``)
is re-embedded with the configured embedder. Because the embedders are
deterministic, re-embedding yields byte-identical vectors -- so the two stores
hold the same ids, chunks, and vectors, and the benchmark compares engines (not
accidental data differences).

The source pgvector store is read-only here; it is never modified, so it stays
intact for benchmarking.

Provider
--------
``RAG_PROVIDER`` must be the real embedder that populated the table (same
provider, same dim) -- the offline ``fake`` embedder has the wrong dim. Override
per run with ``--provider``.

Usage
-----
    python scripts/migrate_pg_to_milvus.py
    python scripts/migrate_pg_to_milvus.py --provider anthropic --batch-size 256
"""

import argparse
import sys
from typing import Any

from industryiq.config import Settings, get_settings
from industryiq.core.embeddings import Embedder
from industryiq.core.milvusvectorstore import MilvusVectorStore
from industryiq.core.pgvectorstore import PgVectorStore


def build_embedder(settings: Settings) -> Embedder:
    """The real embedder for ``settings.provider`` -- must match the pg table's."""
    if settings.provider == "bedrock":
        from industryiq.core.bedrock import BedrockEmbedder

        return BedrockEmbedder(model_id=settings.bedrock_embed_model_id, region=settings.aws_region)
    if settings.provider == "anthropic":
        from industryiq.core.local_embeddings import LocalEmbedder

        return LocalEmbedder()
    raise SystemExit(
        f"provider {settings.provider!r} has no real embedder; migration re-embeds "
        "stored text and must use the same embedder that populated pgvector. "
        "Set RAG_PROVIDER=anthropic (or bedrock)."
    )


def _batches(items: list[Any], size: int) -> list[list[Any]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--provider",
        choices=["anthropic", "bedrock"],
        default=None,
        help="Override RAG_PROVIDER (must match the embedder that populated pgvector).",
    )
    parser.add_argument(
        "--batch-size", type=int, default=128, help="Chunks per embed+upsert batch."
    )
    parser.add_argument("--limit", type=int, default=None, help="Only migrate the first N chunks.")
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Drop the Milvus collection first, so a changed MILVUS_INDEX_TYPE "
        "takes effect (Milvus does not rebuild the index of an existing collection).",
    )
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    settings = get_settings()
    if args.provider:
        settings = Settings(**{**settings.__dict__, "provider": args.provider})
    if not settings.database_url:
        raise SystemExit("DATABASE_URL is not set (the source pgvector store).")

    embedder = build_embedder(settings)
    source = PgVectorStore(settings.database_url, dim=embedder.dim)

    if args.recreate:
        from pymilvus import MilvusClient

        MilvusClient(uri=settings.milvus_uri, token=settings.milvus_token or "").drop_collection(
            settings.milvus_collection
        )
        print(f"dropped existing collection {settings.milvus_collection!r}")

    dest = MilvusVectorStore(
        settings.milvus_uri,
        dim=embedder.dim,
        collection=settings.milvus_collection,
        token=settings.milvus_token,
        index_type=settings.milvus_index_type,
    )

    rows = source.all_items(limit=args.limit or 1_000_000)
    print(
        f"provider={settings.provider} dim={embedder.dim} index={settings.milvus_index_type} "
        f"src=pgvector dest=milvus:{settings.milvus_collection}"
    )
    print(f"read {len(rows)} chunks from pgvector")

    migrated = 0
    skipped = 0
    for batch in _batches(rows, args.batch_size):
        ids: list[str] = []
        texts: list[str] = []
        metas: list[dict[str, Any]] = []
        for cid, meta in batch:
            text = meta.get("text", "")
            if not text:
                skipped += 1
                continue
            ids.append(cid)
            texts.append(text)
            metas.append(meta)
        if not ids:
            continue
        vectors = embedder.embed(texts)
        dest.upsert(ids, vectors, metas)
        migrated += len(ids)
        print(f"  migrated {migrated}/{len(rows)}")

    tail = f", {skipped} skipped (no text)" if skipped else ""
    print(f"\nDone: {migrated} chunks into Milvus{tail}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
