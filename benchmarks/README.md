# Benchmarks

Two complementary benchmarks over the **same** gold query set (`queries.json`):

1. **Retrieval** (`run_benchmark.py`) — *did RAG surface the right chunks?* No LLM,
   cheap and deterministic. Covered first, below.
2. **Chat generation** (`run_chat_benchmark.py`) — *given the chunks it retrieved,
   how good is the chatbot's answer?* An independent LLM judges each answer against
   the retrieved context. Covered in [Chat generation benchmark](#chat-generation-benchmark-llm-as-judge).

They split cleanly: the first scores the *retriever*, the second scores the
*chatbot* by driving the real `ChatService` — a fixed yardstick to run before and
after a backend change (e.g. adding query rewriting).

## RAG retrieval benchmark

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
| `queries.json` | The labeled query set, shared by both benchmarks (see schema below). |
| `metrics.py` | Pure, side-effect-free metric functions (recall@k, precision@k, hit@k, MRR, latency summaries). |
| `run_benchmark.py` | Retrieval runner: read chunks from the DB → resolve gold → embed + search each query → score. |
| `judge.py` | The LLM-as-judge: prompt + structured-output schema + judge client for the chat benchmark. |
| `run_chat_benchmark.py` | Chat runner: drives the real `ChatService` per question → judges the answer against the retrieved context. |

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

Each run prints a `SETUP:` line and records the same block under `config` so a saved
run is self-documenting: `timestamp`, `label`, `provider`, `embedder` + `embed_dim`,
`k`, `queries_file`, `n_queries`, and `n_chunks` (corpus size). The `summary` holds
the recall/latency numbers. The query set and gold set are unchanged between runs, so
any difference in recall or `search_ms` is purely the index method. An exact scan returns true nearest neighbors; an approximate index
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

---

# Chat generation benchmark (LLM-as-judge)

Where the retrieval benchmark asks *"did RAG surface the right chunks?"*, this one
asks the orthogonal question: **given the chunks it retrieved, how good is the
chatbot's answer?** For each question it runs the **real** chat pipeline — the same
`ChatService` the app serves (route → rewrite → retrieve → filter → generate) — then
an **independent judge model** (default `claude-opus-4-8`) scores the answer.

The judge scores **two decoupled axes** so one harness answers both questions you
can ask of a RAG chat — *is the final answer right?* and *did the bot stay faithful
to what it retrieved?* This is what lets you compare a retrieval technique (query
rewriting) **and** isolate the chatbot, from the same run.

Using a separate, stronger judge model — a different API from the chatbot's own —
avoids the self-evaluation bias of a model grading its own output.

## How the judge scores (rubric)

The judge returns a structured verdict per answer (integers 1–5), via the
Anthropic structured-outputs schema in `judge.py`:

- `correctness` — does the answer match the **gold answer** (from `gold_needles`)? Graded on meaning, not wording. **Declining when the gold answer exists scores low** — a miss is a miss. *This is the end-to-end signal a retrieval change moves.*
- `groundedness` — is every claim in the answer supported by the **retrieved context**? (faithfulness / hallucination check; independent of correctness)
- `citation` — did it cite sources as `[n]` against the numbered passages, as instructed?
- `context_sufficient` — *diagnostic only*: did the context actually contain the answer? It must not move the scores; it's used to split the report.

The two axes are deliberately independent: a correct-but-ungrounded answer (high
`correctness`, low `groundedness`) flags the bot answering from outside knowledge
rather than the retrieved context.

## Running it

Needs `DATABASE_URL` (the live store to retrieve from), `RAG_PROVIDER`
(`anthropic`/`bedrock`, the chat model + embedder — same constraint as the
retrieval benchmark), and `ANTHROPIC_API_KEY` for the **judge**, which always calls
the Anthropic API directly regardless of provider. Costs real tokens: one chat turn
(whatever the `ChatService` does) plus one judge completion per question.

```bash
# Score the chatbot as the backend currently behaves.
python benchmarks/run_chat_benchmark.py

# Quick smoke run on the first few questions.
python benchmarks/run_chat_benchmark.py --limit 5
```

Useful flags: `--rewriter {llm,noop}` (the technique under test, see below),
`--label TEXT` / `--out PATH` (tag and save a run), `--judge-model ID`,
`--chat-model ID` (the model under test), `--provider {anthropic,bedrock}`, `--k N`,
`--limit N`, `--queries PATH`.

## Comparing techniques (hot-swappable)

This is what the harness is built for. The chat pipeline is ports-and-adapters, so
techniques are swappable without editing the runner:

- **Query rewriter** — hot-swappable from the CLI via `--rewriter NAME`, which picks
  an entry from the `REWRITERS` registry at the top of `run_chat_benchmark.py`. Two
  ship today: `llm` (production `LlmQueryRewriter`, the default) and `noop` (baseline,
  passes the question through). The choice is written to the run's `config`.
- **Router / relevance filter** — already configurable via `CHAT_ROUTER` and
  `CHAT_RELEVANCE_THRESHOLD` (env), since `build_chat_service` reads them from settings.
- **Anything else** — because the benchmark drives the **real `ChatService`**, any
  backend change not exposed as a flag (a new prompt, a reranker) is reflected too;
  just re-run before and after.

**Add a technique:** implement a `QueryRewriter` adapter in the backend (e.g. in
`core/chat/adapters/rewriting.py`), then add one line to `REWRITERS`:

```python
REWRITERS = {
    "llm": LlmQueryRewriter,
    "noop": lambda _llm: NoOpQueryRewriter(),
    "hyde": lambda llm: HydeQueryRewriter(llm),   # <- your new technique
}
```

It's then selectable as `--rewriter hyde` and recorded in `config`. Hold the judge,
gold set, and chat model fixed across runs and diff the JSON:

```bash
python benchmarks/run_chat_benchmark.py --rewriter noop --label baseline --out baseline.json
python benchmarks/run_chat_benchmark.py --rewriter hyde --label hyde     --out hyde.json
```

> ⚠️ The shipped `LlmQueryRewriter` only condenses **follow-ups**, so on these
> single-turn questions `llm` and `noop` score **identically** — it returns the
> question unchanged with no history. A technique only moves the metrics here if it
> rewrites *standalone* queries (expansion, HyDE, …), which is the usual case for a
> retrieval-side rewrite.

Read the diff on two axes:

- `correctness` / `correctness_pass_rate@4` / `rag_hit_rate` / `context_sufficient_rate`
  **rise** when the change feeds the chatbot better context — this is the change's
  payoff (a better query surfaces the answer more often, so the final answer is right
  more often).
- `correctness_when_context_sufficient` (the bot's extraction skill on the cases where
  the answer *was* retrieved) and `groundedness` are intrinsic to the chat model and
  should stay roughly **flat**. If they move, the judge or chat model changed, not the
  retrieval.

## Output

`--out` writes `{config, summary, rows}`, and a `SETUP:` line with the same `config`
prints at the start of every run. `config` captures the full experiment setup so a
saved run is self-documenting and diffs are unambiguous: `timestamp`, `label`,
`provider`, `chat_model`, `judge_model`, `judge_max_tokens`, `embedder` + `embed_dim`,
`rewriter`, `router`, `relevance_threshold`, `history_turns`, `k`, `queries_file`,
`n_queries`, and `n_chunks` (corpus size in the live store). `summary` holds the
metrics below; `rows` is the per-query detail (scores + the judge's rationale).

## Chat metrics

- `correctness` — mean judge score (1–5) against the gold answer. **The headline
  end-to-end metric.**
- `correctness_pass_rate@4` — fraction of answers scoring `correctness >= 4`.
- `groundedness` / `citation` — mean judge scores (1–5): faithfulness to the context,
  and citation discipline.
- `rag_hit_rate` — fraction of queries where a gold needle landed in the retrieved
  context (the premise check, and the payoff signal for a retrieval-side backend change).
- `context_sufficient_rate` — fraction the judge deemed answerable from the context.
- `correctness_when_context_sufficient` — mean `correctness` restricted to those cases
  (the cleanest read on the chatbot's generation skill, retrieval noise removed).
- `retrieve_ms` / `generate_ms` / `judge_ms` — per-step timing (mean / median / p95
  / min / max), taken from the `ChatService` step timings plus the judge call.
- `errors` — judge calls that failed (refusal/transport); recorded per row and skipped.

## Notes & limitations

- `correctness` uses `gold_needles` as the reference answer. They're verbatim answer
  facts, so they work well as ground truth — but if you reword a query, keep its
  needle a faithful statement of the answer or `correctness` will misjudge.
- A high `correctness` with low `groundedness` means the bot answered from outside
  knowledge, not the retrieved context — worth checking before trusting a
  retrieval-change comparison (the answer was right *despite* retrieval).
- Scores are LLM judgments: not bit-for-bit reproducible. Keep the judge model
  fixed across runs you intend to compare, and prefer the aggregate metrics over
  any single row.
- The runner mirrors `deps.get_chat_service` but swaps in an **in-memory**
  conversation store, so benchmark turns are never written to your database. If you
  add a chat component to `deps`, mirror it in `build_chat_service` to keep the
  benchmark faithful.
- Each question runs **single-turn** (no history), so follow-up condensing isn't
  exercised — this measures one-shot answering.
