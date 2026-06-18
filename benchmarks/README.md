# RAG retrieval benchmark

A small, reproducible benchmark for this project's **retriever**: a set of
queries, each with a known set of relevant ("gold") chunks, run against the live
Postgres + pgvector store. It measures **whether the gold chunks were retrieved**
(recall / hit / MRR) and **how fast** (embed vs. search latency, throughput). No
LLM is involved — runs are cheap, deterministic, and directly comparable across
index methods.

The benchmark reads the corpus chunks straight from the live `chunks` table (the
same data the app serves) — no separate corpus file, no re-embedding. Search is
read-only, so a run never writes to your database.

## What's here

| File | Purpose |
|------|---------|
| `queries.json` | The labeled query set (see schema below). |
| `metrics.py` | Pure, side-effect-free metric functions (recall@k, precision@k, hit@k, MRR, latency summaries). |
| `run_benchmark.py` | The runner: read chunks from the DB → resolve gold → embed + search each query → score. |

## How relevance is labeled

Recall needs ground truth: for each query, *which chunks are relevant?* Hard-coding
chunk indices would break the moment the corpus was re-ingested (chunk ids are
random and regenerated each time), so instead each query lists **`gold_needles`** —
short verbatim phrases that only appear in a relevant chunk. The runner marks a
chunk as gold when its text contains one of those needles, resolving them to chunk
ids. Because the query set ties to *content*, not ids, it keeps working after a
re-ingest.

`queries.json` schema (per query):

```jsonc
{
  "id": "ai-private-investment-2025",     // stable identifier
  "query": "How much did U.S. private ...", // what we send to the retriever
  "category": "AI",                        // expected source industry (for category-hit@1)
  "gold_needles": ["...$285.9 billion..."] // verbatim phrase(s) marking a relevant chunk
}
```

## Running it

Needs `DATABASE_URL` set (the live Postgres store) and `RAG_PROVIDER` set to the
**same provider that populated the table** — its embedder must produce vectors of
the matching dimension or pgvector rejects them. Use the project virtualenv.

```bash
# Default run: query the live pgvector store, score recall + latency.
python benchmarks/run_benchmark.py

# Tag a run and save it, to compare index methods (see below):
python benchmarks/run_benchmark.py --label pg-seqscan --out pg.json
```

Useful flags: `--label TEXT` (names the run in the output), `--provider
{anthropic,bedrock}` (override `RAG_PROVIDER`), `--k N` (top-k, default
`CHAT_RETRIEVAL_K`), `--limit N` (first N queries), `--queries PATH`, `--out PATH`
(write full results as JSON).

### Comparing index methods

This benchmark is built to compare retrieval index methods on the same data. Run
it once per method with a distinct `--label` and `--out`, then diff the JSON files.
For example, a sequential scan vs. an HNSW index:

```bash
# Baseline: no index -> Postgres sequential scan (exact nearest-neighbor).
python benchmarks/run_benchmark.py --label pg-seqscan --out pg-seqscan.json

# Add an approximate index on the chunks table, then re-run:
#   CREATE INDEX ON chunks USING hnsw (embedding vector_cosine_ops);
python benchmarks/run_benchmark.py --label pg-hnsw --out pg-hnsw.json
```

Each file's `config` records the `label`, `provider`, `k`, and `n_chunks`; the
`summary` holds the recall/latency numbers. The query set and gold set are
unchanged between runs, so any difference in recall or `search_ms` is purely the
index method. An exact scan returns true nearest neighbors; an approximate index
(HNSW/IVFFlat) trades a little recall for much lower `search_ms` — exactly the
tradeoff this benchmark surfaces.

## Metrics

- `recall@{1,3,k}` — fraction of a query's gold chunks found in the top-k.
- `hit@{1,3,k}` — did *any* gold chunk land in the top-k (success rate).
- `precision@k` — fraction of the top-k that is gold.
- `mrr` — mean reciprocal rank of the first gold hit (ranking quality).
- `category_hit@1` — did the #1 result come from the expected industry.
- `embed_ms` / `search_ms` / `latency_ms` — per-query timing (mean / median / p95 /
  min / max), with the embed and vector-search steps timed separately so a slow
  embedder can be told apart from a slow store scan.
- `throughput_qps` — `1000 / mean(latency_ms)`, queries served per second at this
  corpus size.

## Notes & limitations

- The corpus is whatever is currently in the `chunks` table. After re-ingesting,
  check that every `gold_needle` still resolves — the runner errors out and names
  the query if a needle no longer matches any chunk's text.
- `search_ms` is the real pgvector latency (distance op + SQL round trip). With no
  index on `chunks` it reflects a sequential scan; add an HNSW/IVFFlat index to
  benchmark approximate search.
