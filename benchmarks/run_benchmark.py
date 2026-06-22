"""Benchmark the industryiq retriever against a labeled gold set.

This is a pure *retrieval* benchmark: a set of queries, each with a known set of
relevant ("gold") chunks, run against a live vector store -- ``--backend pgvector``
(Postgres) or ``--backend milvus``. It measures **whether the gold chunks were
retrieved** (recall / hit / MRR) and **how fast** (embed vs. search latency,
throughput). No LLM is involved -- runs are cheap, deterministic, and directly
comparable across engines and index methods.

What it does
------------
1. Read the corpus chunks (id, text, category) straight from the live pgvector
   table's metadata -- no separate corpus file, no re-embedding.
2. Resolve each query's gold chunks: a chunk is gold when its text contains one
   of the query's ``gold_needles``.
3. For each query: embed it, search the store for the top-k, and score the
   ranked hit list against the gold set. Search is read-only, so the benchmark
   never writes to your database.

Comparing engines / index methods
---------------------------------
Tag each run with ``--label`` (recorded in the JSON ``config``) and ``--out`` to
a file, then diff. To compare pgvector against Milvus, populate both with the
same corpus (scripts/migrate_pg_to_milvus.py) and run once per ``--backend`` --
the query set and gold set are unchanged, so differences in recall / ``search_ms``
are purely the engine. To compare index methods within one engine, rebuild its
index between runs (e.g. a pgvector seqscan vs. HNSW).

Provider
--------
The embedder follows ``RAG_PROVIDER`` from your environment (.env) and must match
the one that populated the table (same provider, same dim) or pgvector rejects
the query vector:
  * ``anthropic`` -- local fastembed embedder (CPU, no network, free).
  * ``bedrock``   -- Titan embeddings on AWS.
Override per run with ``--provider``.

Usage
-----
    python benchmarks/run_benchmark.py
    python benchmarks/run_benchmark.py --backend pgvector --label pg-hnsw --out pg.json
    python benchmarks/run_benchmark.py --backend milvus --label milvus-autoindex --out milvus.json
"""

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import metrics

from industryiq.config import Settings, get_settings
from industryiq.core.embeddings import Embedder
from industryiq.core.pgvectorstore import PgVectorStore
from industryiq.core.vectorstore import VectorStore

HERE = Path(__file__).resolve().parent
DEFAULT_QUERIES = HERE / "queries.json"


# --------------------------------------------------------------------------- #
# Provider wiring (isolated composition root, mirrors api/deps.py)
# --------------------------------------------------------------------------- #
def build_embedder(settings: Settings) -> Embedder:
    """The embedder for ``settings.provider`` -- same choices as the app.

    Must match the embedder that populated the table; the offline ``fake``
    embedder has a different dim, so it is rejected rather than silently
    producing meaningless (and pgvector-incompatible) vectors.
    """
    if settings.provider == "bedrock":
        from industryiq.core.bedrock import BedrockEmbedder

        return BedrockEmbedder(model_id=settings.bedrock_embed_model_id, region=settings.aws_region)
    if settings.provider == "anthropic":
        from industryiq.core.local_embeddings import LocalEmbedder

        return LocalEmbedder()
    raise SystemExit(
        f"provider {settings.provider!r} has no real embedder; the benchmark queries "
        "live pgvector and needs query vectors that match the stored ones. "
        "Set RAG_PROVIDER=anthropic (or bedrock)."
    )


# --------------------------------------------------------------------------- #
# Corpus loading + gold resolution
# --------------------------------------------------------------------------- #
@dataclass
class Corpus:
    """A queryable store plus the lookup tables the scorer needs.

    ``embedder``/``store`` are exposed so the benchmark can time the embed and
    vector-search steps separately instead of through one combined call.
    """

    embedder: Embedder
    store: VectorStore
    chunk_text_by_id: dict[str, str]
    chunk_category_by_id: dict[str, str]
    n_chunks: int


def build_store(settings: Settings, backend: str, dim: int) -> VectorStore:
    """The live store to benchmark, selected by ``backend``.

    ``pgvector`` and ``milvus`` are both real, persistent engines holding the same
    corpus (see scripts/migrate_pg_to_milvus.py); pointing the *same* queries and
    gold set at each is how the two are compared. The Milvus import is deferred so
    pgvector-only runs don't need pymilvus.
    """
    if backend == "milvus":
        from industryiq.core.milvusvectorstore import MilvusVectorStore

        return MilvusVectorStore(
            settings.milvus_uri,
            dim=dim,
            collection=settings.milvus_collection,
            token=settings.milvus_token,
            index_type=settings.milvus_index_type,
        )
    if not settings.database_url:
        raise SystemExit("DATABASE_URL is not set (the live Postgres store to benchmark).")
    return PgVectorStore(settings.database_url, dim=dim)


def build_corpus(embedder: Embedder, settings: Settings, backend: str) -> Corpus:
    """Benchmark against the *live* vector store (pgvector or Milvus) -- the real path.

    Self-contained against the store: the gold labels (chunk text + category) are
    read straight from the store's own metadata via ``all_items()``, so no
    separate corpus file is needed. Search is read-only, so the benchmark never
    writes to your store, and ``search_ms`` measures the real distance computation
    plus the network round trip.
    """
    store = build_store(settings, backend, embedder.dim)
    text_by_id: dict[str, str] = {}
    category_by_id: dict[str, str] = {}
    for cid, meta in store.all_items(limit=1_000_000):
        text_by_id[cid] = meta.get("text", "")
        category_by_id[cid] = meta.get("category", "uncategorized")
    return Corpus(embedder, store, text_by_id, category_by_id, len(text_by_id))


def resolve_gold(corpus: Corpus, needles: list[str]) -> set[str]:
    """The set of chunk ids whose text contains any of ``needles`` (case-insensitive)."""
    lowered = [n.lower() for n in needles]
    return {
        cid
        for cid, text in corpus.chunk_text_by_id.items()
        if any(n in text.lower() for n in lowered)
    }


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
@dataclass
class Record:
    """Raw per-query scoring inputs, kept un-aggregated so a summary can compute
    any cutoff (recall@1 vs recall@5) without re-running retrieval."""

    id: str
    gold: set[str]
    hit_ids: list[str]
    expected_category: str | None
    top_category: str | None
    latency_ms: float
    embed_ms: float
    search_ms: float


@dataclass
class EvalOutput:
    """Everything one evaluation pass produced."""

    records: list[Record] = field(default_factory=list)
    rows: list[dict[str, Any]] = field(default_factory=list)  # per-query display rows


def cutoffs(k: int) -> list[int]:
    """The k-cutoffs to report recall/hit at -- always 1 and 3, plus the run's k."""
    return sorted({1, 3, k})


def evaluate_retriever(
    corpus: Corpus, queries: list[dict[str, Any]], gold_by_id: dict[str, set[str]], k: int
) -> EvalOutput:
    """Embed and search each query, scoring the ranked hits against its gold set.

    Times the embed and vector-search steps separately (calling the embedder and
    store directly) so slow embedding can be told apart from slow search -- the
    two have very different fixes (batch/cache the embedder vs. index the store).
    """
    out = EvalOutput()
    for q in queries:
        gold = gold_by_id[q["id"]]
        embed_start = time.perf_counter()
        query_vector = corpus.embedder.embed([q["query"]])[0]
        embed_ms = (time.perf_counter() - embed_start) * 1000
        search_start = time.perf_counter()
        hits = corpus.store.search(query_vector, k=k)
        search_ms = (time.perf_counter() - search_start) * 1000
        latency = embed_ms + search_ms
        hit_ids = [h.id for h in hits]
        top_category = corpus.chunk_category_by_id.get(hits[0].id) if hits else None
        out.records.append(
            Record(
                q["id"],
                gold,
                hit_ids,
                q["category"],
                top_category,
                latency_ms=latency,
                embed_ms=embed_ms,
                search_ms=search_ms,
            )
        )
        out.rows.append(
            {
                "id": q["id"],
                f"recall@{k}": round(metrics.recall_at_k(hit_ids, gold, k), 3),
                "hit@1": metrics.hit_at_k(hit_ids, gold, 1),
                "rr": round(metrics.reciprocal_rank(hit_ids, gold), 3),
                "top_category": top_category,
                "embed_ms": round(embed_ms, 2),
                "search_ms": round(search_ms, 2),
                "latency_ms": round(latency, 2),
            }
        )
    return out


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _round_stats(stats: dict[str, float]) -> dict[str, float]:
    return {key: round(value, 2) for key, value in stats.items()}


def _multi_k(records: list[Record], ks: list[int]) -> dict[str, float]:
    """recall@kk and hit@kk averaged over ``records`` for each cutoff ``kk``."""
    out: dict[str, float] = {}
    for kk in ks:
        out[f"recall@{kk}"] = round(
            _mean([metrics.recall_at_k(r.hit_ids, r.gold, kk) for r in records]), 3
        )
        out[f"hit@{kk}"] = round(
            _mean([1.0 if metrics.hit_at_k(r.hit_ids, r.gold, kk) else 0.0 for r in records]), 3
        )
    return out


def retriever_summary(out: EvalOutput, k: int) -> dict[str, Any]:
    recs = out.records
    summary: dict[str, Any] = {"queries_scored": len(recs)}
    summary.update(_multi_k(recs, cutoffs(k)))
    summary[f"precision@{k}"] = round(
        _mean([metrics.precision_at_k(r.hit_ids, r.gold, k) for r in recs]), 3
    )
    summary["mrr"] = round(_mean([metrics.reciprocal_rank(r.hit_ids, r.gold) for r in recs]), 3)
    summary["category_hit@1"] = round(
        _mean([1.0 if r.top_category == r.expected_category else 0.0 for r in recs]), 3
    )
    summary["embed_ms"] = _round_stats(metrics.summarize([r.embed_ms for r in recs]))
    summary["search_ms"] = _round_stats(metrics.summarize([r.search_ms for r in recs]))
    summary["latency_ms"] = _round_stats(metrics.summarize([r.latency_ms for r in recs]))
    mean_latency = _mean([r.latency_ms for r in recs])
    summary["throughput_qps"] = round(1000 / mean_latency, 2) if mean_latency else 0.0
    return summary


def print_section(title: str, summary: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")
    for row in rows:
        print("  " + json.dumps(row, ensure_ascii=False))
    print("  " + "-" * 68)
    print("  SUMMARY: " + json.dumps(summary, ensure_ascii=False))


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--label",
        default=None,
        help="Free-text tag recorded in the output config, to name an index method "
        "when comparing runs (e.g. 'pg-hnsw', 'pg-seqscan').",
    )
    parser.add_argument("--queries", type=Path, default=DEFAULT_QUERIES)
    parser.add_argument(
        "--backend",
        choices=["pgvector", "milvus"],
        default="pgvector",
        help="Which live vector store to benchmark (default: pgvector).",
    )
    parser.add_argument(
        "--provider",
        choices=["anthropic", "bedrock"],
        default=None,
        help="Override RAG_PROVIDER for this run (must match the table's embedder).",
    )
    parser.add_argument(
        "--k", type=int, default=None, help="Top-k to retrieve (default: CHAT_RETRIEVAL_K)."
    )
    parser.add_argument("--limit", type=int, default=None, help="Only run the first N queries.")
    parser.add_argument(
        "--out", type=Path, default=None, help="Write full results as JSON to this path."
    )
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    settings = get_settings()
    if args.provider:
        settings = Settings(**{**settings.__dict__, "provider": args.provider})
    k = args.k or settings.chat_retrieval_k

    spec = json.loads(args.queries.read_text(encoding="utf-8"))
    queries = spec["queries"]
    if args.limit is not None:
        queries = queries[: args.limit]

    print(f"provider={settings.provider}  backend={args.backend}  k={k}  queries={len(queries)}")

    embedder = build_embedder(settings)
    corpus = build_corpus(embedder, settings, args.backend)
    print(f"loaded {corpus.n_chunks} chunks from the live {args.backend} store")

    # Full experiment setup, recorded with the results so each run is self-documenting
    # and reproducible (and diffs across index methods are unambiguous).
    config: dict[str, Any] = {
        "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
        "label": args.label,
        "backend": args.backend,
        # The Milvus index that produced these numbers (None for pgvector).
        "milvus_index_type": settings.milvus_index_type if args.backend == "milvus" else None,
        "provider": settings.provider,
        "embedder": type(embedder).__name__,
        "embed_dim": embedder.dim,
        "k": k,
        "queries_file": args.queries.name,
        "n_queries": len(queries),
        "n_chunks": corpus.n_chunks,
    }
    print("SETUP: " + json.dumps(config, ensure_ascii=False))

    # Resolve + validate gold for every query up front.
    gold_by_id: dict[str, set[str]] = {}
    missing: list[str] = []
    for q in queries:
        gold = resolve_gold(corpus, q["gold_needles"])
        gold_by_id[q["id"]] = gold
        if not gold:
            missing.append(q["id"])
    if missing:
        raise SystemExit(
            "no chunk matched the gold_needles for: "
            + ", ".join(missing)
            + "\nFix the needle text in queries.json so it is a verbatim phrase in the corpus."
        )

    out = evaluate_retriever(corpus, queries, gold_by_id, k)
    summary = retriever_summary(out, k)
    print_section("RETRIEVER", summary, out.rows)

    results: dict[str, Any] = {"config": config, "summary": summary, "rows": out.rows}
    if args.out:
        args.out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nwrote {args.out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
