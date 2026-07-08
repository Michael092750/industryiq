# Chunk sizing & the retrieval length filter

**Status:** in progress — coalescing implemented and applied; the objective benchmark can't
yet adjudicate it (see "Why the benchmark can't decide this"). Current corpus state and the
open decision are at the end.

This documents a chain of decisions about **short chunks** in the retrieval pipeline: why
they hurt, the query-time band-aid we shipped, the measured tuning, and the ingest-time
chunker fix (coalescing) that the band-aid was covering for. It's the sibling of
[figure-ingestion.md](figure-ingestion.md) — the figure backfill is what made this acute.

## The problem: short chunks

The retriever returns the top-k (k=5) chunks by store score (Milvus RRF dense+BM25 / pgvector
cosine). **Short chunks systematically score high but answer nothing:** a bare heading
(`## U.S. AI Private Investment`), a one-line figure caption (`Figure: a stethoscope on a
microchip`), or a small figure table is *pure topic, no dilution*, so its embedding sits very
close to the (also short, also topical) query — closer than the paragraph that actually
contains the answer, whose signal is diluted across several sentences. BM25's term-frequency
normalization favors short chunks too. So short chunks sweep top-k slots away from real
answers.

The [figure backfill](figure-ingestion.md) made this worse: it injected ~thousands of
one-line `Figure: …` descriptions and small figure tables as standalone chunks. On the
post-backfill corpus, **42% of chunks were under 400 chars.**

## Two (three) independent mechanisms — don't conflate them

| Mechanism | Stage | What it guarantees |
|---|---|---|
| **Atomic chunking** (`chunk_markdown`) | ingest | *Completeness* — a GFM table / fenced block is one chunk, **never split**. |
| **`min_chunk_chars` filter** (`Retriever`) | query | *Substance* — over-fetch k×6, **drop** chunks under the floor, take top-k. |
| **`chunk_min_chars` coalescing** (`chunk_markdown`) | ingest | *Substance* — **merge** small pieces into neighbours so few short chunks exist. |

A chunk can be **whole and tiny**. Atomic chunking stops *fragmentation* (a half-table is
useless); it does nothing about *short-but-complete* chunks. Those are handled either by
dropping them at query time (the filter) or merging them at ingest (coalescing).

## Decision 1 — the query-time filter, and its measured optimum

The filter (`RETRIEVAL_MIN_CHUNK_CHARS`, `Retriever.retrieve`) over-fetches then removes
sub-floor hits, **preserving store rank order** (it's a filter-and-slide, not a re-rank):
substantial chunks below dropped short ones slide up. Fallback to raw hits only if *all*
candidates are short.

We swept it on the figure-backfilled corpus (Milvus, 42-query subset — see caveats):

| min_chunk_chars | recall@1 | recall@3 | recall@5 | hit@1 | mrr |
|---|---|---|---|---|---|
| 0 | 0.333 | 0.464 | 0.655 | 0.357 | 0.457 |
| 200 (old default) | 0.345 | 0.548 | 0.714 | 0.381 | 0.501 |
| **400** | **0.405** | 0.583 | 0.714 | **0.452** | **0.549** |
| 500 | 0.381 | 0.583 | 0.738 | 0.429 | 0.543 |
| 800 | 0.357 | 0.595 | 0.714 | 0.405 | 0.523 |

**Counterintuitive result: raise it, don't relax it.** mrr / recall@1 / hit@1 peak at **400**
and decline past 500. Relaxing toward 0 was worst on every metric. The figure pass added many
short chunks, so a *stronger* short-chunk filter helps more, not less. → **default raised
200 → 400.**

Cost of the filter, though: it *drops* content, including **29% of the real figure tables**
(the ones under 400 chars) — the very data the backfill paid to transcribe. It also can't
adjudicate its own worth on this benchmark (below). That motivated the real fix.

## Decision 2 — the real fix is at ingest: coalescing (merge, don't drop)

The filter is a query-time band-aid for an **ingest-time chunking problem**: the chunker emits
orphan short chunks (a lone small table, a caption between two figures, a short trailing
remainder). The fix is to **merge** small pieces into a substantial neighbour at ingest, so
few short chunks exist — merging *keeps* the content (a small table rides *with* its context,
better embedding + still retrievable) where the filter throws it away.

`chunk_min_chars` (new setting, default 400) enables `_coalesce_small` in `chunk_markdown`.
**Two variants were tried:**

- **Greedy** (v1) — accumulate any pieces until the floor is reached. Simple, but it prepends
  a stray table/caption onto the *front* of a large prose chunk, **diluting** that prose.
- **Non-diluting** (v2, current) — only merge pieces that are *themselves* under the floor,
  and only with each other; a piece already at/over the floor is emitted untouched. A run of
  small tables/captions coalesces into one block; a full paragraph is never diluted.

Both **never split a table** (pieces merge whole).

### Measured result (Milvus, 42-query subset, best filter mcc=400)

| corpus | chunks | <400 | recall@1 | recall@5 | mrr | figures retrievable? |
|---|---|---|---|---|---|---|
| non-coalesced + filter | 30,164 | 42% | **0.405** | **0.714** | **0.549** | ❌ small tables dropped by filter |
| greedy coalesce | 22,348 | 19% | 0.345 | 0.690 | 0.498 | ~ absorbed into prose chunks |
| non-diluting coalesce | 28,454 | 36% | 0.357 | 0.631 | 0.478 | ✅ standalone figure blocks |

The non-diluting v2 was **worse**, not better — disproving the "dilution was the culprit"
hypothesis. The prose numbers rank **inversely to how retrievable figures are**: surfacing a
figure run as a standalone block makes it *survive the filter and compete for top-5 slots*,
and this benchmark's gold is 100% prose, so it scores that as pure loss.

## Why the benchmark can't decide this

`benchmarks/queries.json` has **50 queries, all with prose-answer gold** (the resolvable 42
after excluding 8 stale needles). It rewards the filter's "keep prose focused, hide short
noise" and **structurally cannot credit** the thing coalescing is *for* — keeping small
**figure tables** retrievable. It sees coalescing's cost (prose competition) and none of its
benefit. **Optimizing this benchmark ⇒ hide figures ⇒ waste the ~$8.6 figure backfill.** So it
is the wrong compass for this decision.

The only fix is to **measure the goal**: add figure-answer queries. The 8 needles that stopped
resolving are the seed for this (see next section).

## The 8 stale gold needles

The needles were authored against the *old pypdf* corpus; two docling re-ingests later, the
verbatim strings drifted, so they no longer resolve (and the runner hard-fails if they're
included). Three causes:

- **A. pypdf extraction artifact** — e.g. `'reaching 17 .1 million H100-equivalents'`; the
  `17 .1` space is pypdf's; docling has clean `17.1 million` (present in the corpus). *Fixable
  by rewriting the needle.*
- **B. rephrased / different reading order** — `"79 acquisitions"`, `"17.6 percent"`,
  discount-window/`$150 billion`, Stargate, life-insurance: the fact is present, the exact
  sentence differs. *Fixable.*
- **C. genuinely missing** — (1) Mira Murati's "Thinking Machines" seed round: the **source
  document isn't in the library**. (2) The `"29.4 percent"` variable-rate-debt figure: a
  **chart value the backfill didn't recover** (see below). *Not fixable by editing needles.*

**6/8 are benchmark rot (re-point the needle); 2/8 are real content gaps.**

### Resolution (2026-07-06) — verified against the live corpus AND the pypdf baseline

Cross-checking each of the 8 stale needles against **both** the live Docling corpus (28,454
chunks) and the 6/19 **pypdf baseline** (`ragbenchresults/baseline_pypdf_pgvector/rag_baseline.json`)
corrected the split. The earlier "6 rot / 2 gaps" *and* the regression doc's "all 11 dropped
needles are content loss" were both off. Verified reality: **7 are recoverable (measurement
artifacts), 4 are genuine Docling content-loss regressions.** The source PDFs are unchanged, so a
fact that resolved under pypdf but not Docling was *dropped by the parser*, not missing from the
library.

- **Recoverable — fact IS in the Docling corpus, only the pypdf-verbatim needle drifted** (this is
  the gold-set bias of §3.2, and it accounts for 7/11, not a minority):
  - Auto-fixed by the normalizer, no edit — `ai-compute-capacity` (`17 .1`→`17.1` spacing),
    `ai-patent-top-offices-share`, `ai-pct-patent-growth`, `agriculture-ecoceres-round` (spacing/
    punctuation), and `healthcare-biopharma-rd-share` (present as `R&amp;D` — Docling emits
    **HTML-escaped** markdown; folded by `html.unescape`).
  - Re-anchored to the Docling text — `ai-field-acquisitions` → `"79 acquisitions in total"` (the
    `(18 percent…)` parenthetical was reworded away); `finance-discount-window-credit` →
    `"all-time high of 153 billion"` (Docling states it as a `$153 billion` all-time high; the old
    `$5B→$150B` was pypdf's rendering).
- **Genuine Docling content-loss regressions — present AND retrieved under pypdf, gone now**
  (marked `expected_missing`, to be *recovered at ingest*, not accepted as gaps):
  - `finance-variable-rate-debt-share` — pypdf **hit@1=True** (read `29.4%` as flat text); Docling
    routed it to a figure, so `variable rate debt` survives only as a table header + `Figure 2.4`
    caption, the value gone (`gold_type: figure`; becomes the first figure-gold once recovered).
  - `ai-thinking-machines-seed` — pypdf recall@5=1.0/hit@1; Docling dropped the Thinking Machines
    Lab content (only Alan Turing's 1950 "thinking machines" remains).
  - `finance-life-insurance-surrender-risk` — pypdf recall@5=1.0; Docling dropped the quantified
    Moody's `$500B/one-third/low-penalty` claim (only qualitative surrender risk remains).
  - `ai-stargate-investment` — pypdf recall@5=0.333; Docling dropped the `$100B–$500B` build figure
    (only qualitative Stargate mentions + a Forbes citation title remain).

**Blast radius (full pypdf↔Docling diff, pgvector dense):** hit@1 0.40→0.256, recall@5 0.585→0.466,
mrr 0.515→0.359. Beyond the 4 content-loss needles above, **7 more queries in the shared set
regressed on *ranking*** (gold chunk still present but no longer in top-5, e.g.
`finance-noninterest-income`, `healthcare-biopharma-output-2022`, `semi-asia-manufacturing-share`) —
the chunking/filter symptom of §4, distinct from content loss. So Docling's regression is: **4
facts lost at parse + ~7 facts ranked out + 7 measurement artifacts that were never really lost.**

**Mechanism added** so this no longer hard-blocks a run: gold resolution is now
whitespace/punctuation/HTML-entity-insensitive (`benchmarks/textmatch.py`, shared by both
benchmarks), unresolved-but-real needles print a **hybrid-search re-anchor hint** (nearest current
chunk text to copy from) instead of a bare error, and `expected_missing: true` tracks a
**regression pending an ingest fix** without failing the run or being scored (the note on each says
"recover at ingest, then clear the flag"). Queries also carry `gold_type` (`prose`|`figure`) and the
retrieval summary **splits recall/mrr by it** once a figure query exists. Post-fix baseline: 46
scored, 4 expected_missing, green.

### Why a chart value can go missing (the 29.4% case)

Docling deletes a chart's data at parse (it becomes a bare `<!-- image -->`); the figure pass
is the **only** recovery path, with three gates — **detection** (docling tags it a picture and
emits a crop), **size filter** (crop ≥ `min_pixels`: 200 collect / 300 submit), **transcription**
(the VLM reads the value). Fail any gate → the number is gone with no fallback.

**Traced to ground truth (2026-07-06).** The 29.4% fact is a **chart footnote**, not a plotted
value: `imf_org__text_587835eb.pdf` p79 — *"⁶For a sample of 518 North American and 157 European
high-yield corporate bond issuers, the average share of variable rate debt is 29.4 percent…
Sources: S&P Capital IQ."* pypdf extracted that footnote as flat text (so the 6/19 baseline had it,
**hit@1**); in the Docling corpus that doc has **345 chunks, 0 containing `29.4`, 0 mentioning
"variable rate debt"** — the whole figure region (chart + its footnote/source note) was rasterized
to `<!-- image -->` and dropped. So it isn't a `min_pixels` miss, and it isn't quite a
"transcription miss" either: the VLM prompt asks for the chart's **data table**, so even a perfect
transcription would skip a **footnote/source-note**. This generalizes — the losses are
*figure-attached prose* (footnotes, annotations, average-line labels, timeline entries, boxed
callouts), which the data-table prompt never captures. `min_pixels` is *not* the lever.

**Fix shipped (2026-07-06): hybrid pypdf text recovery, default on.** Lever (1) —
`PDF_HYBRID_RECOVERY` (default `1`) in `loaders._recover_dropped_pages` — appends, per page, the
pypdf text-layer lines Docling dropped (deduped against the Docling text and running
headers/footers). Free, no VLM, lossless for text-layer content; it recovers all four content-loss
regressions (`29.4%`, Stargate, Thinking Machines, life-insurance). Validated end-to-end: Docling
alone drops `29.4 percent` from `587835eb`, recovery restores it on p79. See
[figure-ingestion.md](figure-ingestion.md#hybrid-text-recovery-the-completeness-net). Secondary
levers not yet needed: widening the figure prompt to include in-figure footnotes; Sonnet over Haiku
for dense charts. **Landed in the corpus 2026-07-07** (both stores, all four facts recovered) — but
not without incident: the bulk batch ingest had a *parallel* Docling parse that silently bypassed
the net, and OCR-on had degraded far more than the four facts. The full story, decisions, and the
OCR-off recommendation are in [hybrid-recovery-reload.md](hybrid-recovery-reload.md).

## Changes added

| File | Change |
|---|---|
| `config.py` | `retrieval_min_chunk_chars` 200 → **400**; new **`chunk_min_chars=400`** (+ `CHUNK_MIN_CHARS` env). |
| `core/chunking.py` | `_coalesce_small` + `chunk_markdown(min_chars=…)` — non-diluting merge of sub-floor pieces; tables kept whole. |
| `core/pipeline.py` | `RagPipeline(chunk_min_chars=…)`, passed into `chunk_markdown` in `ingest_pages`. |
| `api/deps.py` | `get_pipeline` passes `chunk_min_chars=settings.chunk_min_chars`. |
| `core/figure_batch.py` | `reset_for_rechunk` — rewind a finished job to the write phase to re-apply chunking from saved plans (no re-parse/re-transcribe/API cost). |
| `scripts/ingest_bulk_batch.py` | `--rechunk` flag. |
| `tests/` | `test_chunking.py` coalescing tests (non-dilution, run-merge, table-stays-whole, whole-text-under-floor); `test_figure_batch.py` rechunk-adjacent. |

Re-chunking the whole corpus is cheap: `ingest_bulk_batch.py --reprocess-all --rechunk
--work-dir <dir>` re-fetches the batch results (available ~29 days) and re-chunks + re-embeds +
re-writes each doc with the current chunker — **no PDF re-parse, no figure re-transcription, $0.**

## Current state & open decision

- **Corpus (local both stores):** currently **non-diluting coalesce** (28,454 chunks, parity),
  which is the **worst** of the three on the prose benchmark. Not a good resting state.
- **Config defaults:** `retrieval_min_chunk_chars=400`, `chunk_min_chars=400`.

**Open decisions (blocked on a fair eval):**
1. **Which corpus** — revert to **greedy coalesce** (best figure/prose compromise, −0.05 mrr,
   figures retrievable via prose chunks) or **non-coalesced + filter@400** (best prose, figures
   hidden). A one-command `--rechunk` either way.
2. **Build a figure-answer eval** — *infra done (2026-07-06):* fixable needles re-anchored, gaps
   flagged `expected_missing`, and `gold_type` + split reporting are live, so the moment a `figure`
   query exists the summary shows figure-vs-prose recall separately. **Remaining:** author the
   chart-fact queries themselves (seed from figure tables the VLM *did* transcribe; the recovered
   `29.4%` becomes the first). This is what finally lets the benchmark credit surfacing figures.
3. **Chart coverage** — chase the 29.4%-class misses; likely Sonnet-vs-Haiku on chart-dense
   financial docs, not `min_pixels`.
