# Figure ingestion — recovering chart/figure content at ingest

**Status:** implemented, behind a flag (`FIGURE_VLM`, default `off`); pending a full
re-ingest to land in the corpus.
**Decision:** Option B — transcribe figures with an off-box vision model (Claude) in a
**separate pass over docling's finished document**, not inside docling's pipeline.

**Companion fix (2026-07-06):** figure transcription only recovers a chart's *data table*.
Figure-*attached prose* — chart footnotes, source notes, average-line labels, timeline
entries, boxed callouts — is dropped by the same picture handling and the VLM prompt never
asks for it. That is recovered separately by **hybrid pypdf text recovery** (default **on**,
`PDF_HYBRID_RECOVERY`), documented in [Hybrid text recovery](#hybrid-text-recovery-the-completeness-net)
below. The two are complementary: the VLM reads the pixels a chart plots; pypdf reads the
text layer around it.

## Problem

Docling's layout model reconstructs prose reading order well, but it classifies charts
and figures as *picture* regions and **drops their content** from the Markdown export —
each figure becomes a bare `<!-- image -->` placeholder. Our benchmark analysis
(`ragbenchresults/.../docling_pgvector_regression_analysis.md`) traced the retrieval
regression vs the old pypdf baseline to exactly this: the queries that stopped resolving
were overwhelmingly **numeric facts that live inside charts** (`"17.1 million
H100-equivalents"`, `"29.4 percent variable rate debt"`, `"$400 million round"`). pypdf
dumped those as flat text; docling omits them.

**Two distinct kinds of dropped content** (they need different fixes — see the 2026-07-06
diagnosis in [chunk-sizing-and-retrieval-filter.md](chunk-sizing-and-retrieval-filter.md)):

1. **A chart's plotted data** — bar heights, series values, axis ticks. Only readable from
   the *pixels*. Recovered by the **figure-VLM pass** (below), which re-tabulates the chart.
2. **Figure-attached prose** — footnotes, source notes, average-line labels, timeline
   entries, boxed-callout text. This is *text-layer* content pypdf reads fine; the VLM's
   "output the data table" prompt structurally skips it. Recovered by **hybrid pypdf text
   recovery** (below). Verified case: the `29.4%` variable-rate-debt fact is a chart footnote
   in `imf_org__text_587835eb.pdf` p79 (*"…the average share of variable rate debt is 29.4
   percent … Sources: S&P Capital IQ"*); docling drops it (that doc had 345 corpus chunks, 0
   with `29.4`), and hybrid recovery restores it on the correct page.

## Options considered

| | How | Verdict |
|---|---|---|
| **A. Docling-native API picture description** | `do_picture_description` + `PictureDescriptionApiOptions` → an OpenAI-compatible VLM endpoint | ❌ Rejected |
| **B. Custom Claude-vision pass over the finished document** | docling exports figure images; we call Claude per figure and splice the result back in | ✅ **Chosen** |
| **C. Send whole chart-heavy pages to Claude** | skip figure plumbing; hand Claude the PDF page | ❌ Rejected |

### Why not A
- **Claude is not OpenAI-compatible** — docling's API option speaks the OpenAI
  chat/completions shape, so using Claude would need a LiteLLM-style proxy (extra infra).
- **It can't recover the numbers anyway.** The API option covers *picture description*
  only. **Chart extraction (`do_chart_extraction`) is local-Granite-only — there is no API
  backend** (`chart_extraction_options.model` is `ChartExtractionModelKind.GRANITE_VISION_V4`
  with no `url` field). The stage that actually produces chart→CSV data (`PictureTabularChartData`
  etc.) can only run the local model.
- **It runs inside the fragile `StandardPdfPipeline`.** A per-figure network call inside
  that loop is one more way for the whole document's parse to fail — and on failure our
  loader silently falls the entire document back to pypdf (`loaders.py`), so the corpus
  degrades invisibly.

### Why not C
- Most expensive (whole pages, including pages with little chart content), fuzzy
  attribution (free-form page facts don't map to chunks; risk of duplicating prose docling
  already has), and higher hallucination risk than transcribing a bounded figure image.

### Why B
- **Provider-native** — uses our existing `ANTHROPIC_API_KEY`; no OpenAI, no proxy.
- **Replaces both broken local stages with one call.** One Claude call per figure does
  what chart extraction *and* picture description do (prompt: chart → Markdown table of
  values; other figure → one-line description), so we keep `do_chart_extraction=False`
  and `do_picture_description=False`.
- **Decoupled from the fragile pipeline.** The call runs *after* conversion, so a VLM
  failure loses one figure, not the document. We also isolate each figure in a `try`.
- **Cheapest per unit + filterable** (skip tiny logos/icons; cap for test runs) and
  **reprocessable** (re-run the figure pass without re-parsing PDFs, if outputs are cached).

## Environment findings that forced this (why the docling-native paths don't run here)

Enabling docling's own local vision stages crashed the pipeline on this machine:
1. **`torch.compile` needs MSVC `cl.exe`** (not installed on Windows) → `InductorError:
   Compiler: cl is not found`. Softened via `_soften_torch_compile()` (falls back to eager),
   but then:
2. **transformers 5.12 breaks the vision models' forward pass** → `TypeError:
   create_causal_mask() got an unexpected keyword argument 'cache_position'`. Docling's
   Granite/SmolVLM code expects transformers 4.x. Fixing this means a risky transformers
   downgrade in an isolated ingest env.

So docling's local chart-extraction / picture-description are effectively unavailable here.
Option B sidesteps both — no local vision models at all.

## How it works

`src/industryiq/core/figure_vlm.py`, wired into `loaders._load_pdf_pages_docling`:

1. **Convert** the PDF with docling (enrichment stages OFF; `generate_picture_images=True`,
   `images_scale=2` so cropped figure images are available and memory stays modest).
2. **Location comes from docling for free.** Each figure is a `PictureItem` carrying its
   `prov` (`page_no` + `bbox`) and reading-order position — we operate on docling's own
   structured output, not a blind re-parse. `pic.get_image(doc)` gives the cropped image.
3. **Transcribe** each figure with a provider-agnostic `Annotator` (Claude vision by
   default). One strict prompt handles **all figure types**: data chart / table-image →
   GFM table of values (with units); diagram/photo/logo → one-line `Figure: …` description.
4. **Inject** each result into the page Markdown in place of its `<!-- image -->`
   placeholder (`inject_figures`). Placeholders and figures are both in reading order, so
   the k-th placeholder gets the k-th figure. On any count mismatch we strip placeholders
   and append the figures at the end of the page — content is never lost or misplaced.

The pass is provider-agnostic (`annotate_document_figures` takes any `image -> str`
callable), so it's unit-tested offline with a fake and swappable per provider.

## Hybrid text recovery (the completeness net)

`src/industryiq/core/loaders.py` (`_recover_dropped_pages`), default **on**
(`PDF_HYBRID_RECOVERY`). It is wired into **both** Docling parse paths — the synchronous
loader (`_load_pdf_pages_docling`) *and* the batch collector
(`figure_batch._docling_pages_and_doc`), which has its own parallel parse; both must apply
it or a bulk re-ingest silently loses the recovered text (locked by
`tests/test_figure_batch.py::test_docling_pages_and_doc_applies_hybrid_recovery`). For each
page it appends the **pypdf lines Docling dropped** — i.e. a
line pypdf's flat text-layer extraction has that isn't already in the Docling Markdown. That
set is precisely the figure-attached prose Docling rasterized away (footnotes, source notes,
annotations, timeline entries, callouts). It runs *last*, so it sees the VLM injections and
won't re-add anything they already restored.

How it stays clean rather than dumping all of pypdf's messy text back in:

- **Dedup is whitespace/punctuation/entity-insensitive** (fold to lowercase alphanumerics),
  so any prose Docling kept — even reflowed into a table cell or respaced (`"17 .1"` vs
  `"17.1"`) — is recognized as already present and *not* re-added. Docling keeps ownership of
  reading order; recovery only adds what's genuinely missing.
- **Prose, not digit-soup.** A line needs ≥ 10 letters to qualify, so numeric table rows and
  axis-tick columns are skipped — those are the figure-VLM's job, not text recovery.
- **No running headers/footers.** A line repeated on ≥ 3 pages *and* ≥ 25% of pages is treated
  as a page banner and dropped, so recovery doesn't staple a header onto every page.
- **Page attribution preserved.** pypdf pages align 1:1 to Docling pages when the counts match
  (recovered text keeps its page number for citations); on a mismatch it recovers against the
  whole document and appends once, so content is never lost even if attribution is imperfect.
- **Best-effort, never fatal.** Any pypdf failure logs and returns the Docling pages unchanged
  — the safety net can't take down a parse that already succeeded. (Bonus: on pages where
  Docling itself OOMs (`std::bad_alloc`) and emits nothing, pypdf's text for that page is all
  "dropped" and thus fully recovered — so hybrid also backstops Docling's memory failures.)

Cost is negligible — one extra pypdf parse per document (sub-second), no tokens. Unlike the
VLM pass it is **free and lossless for text-layer content**, which is why it is on by default
while `FIGURE_VLM` stays opt-in. Unit-tested in `tests/test_hybrid_recovery.py`.

> **Landing it in the corpus (2026-07-07) surfaced two issues** — the batch ingest had a
> parallel Docling parse that silently bypassed this net, and OCR-on had degraded far more than
> the targeted footnotes. The incident, decisions (migrate + rechunk to avoid a re-charge), and
> the OCR-off recommendation are recorded in
> [hybrid-recovery-reload.md](hybrid-recovery-reload.md).

## Chunking compatibility

The transcribed content is injected **upstream of chunking**, so placement into the right
page/section chunk is automatic. Charts come back as **GFM tables**, and
`chunk_markdown` keeps GFM tables (and fenced blocks) **atomic** — a table is emitted as
its own chunk and **never split across a boundary**, even if it exceeds `chunk_size`.
`tests/test_figure_vlm.py::test_injected_table_is_kept_atomic_by_chunk_markdown` locks
this in. Each injected figure is wrapped in blank lines so it reads as a standalone block.

Small transcribed tables/captions can still land as **short** chunks, which interacts with
retrieval ranking — how that's handled (the `min_chunk_chars` filter vs. ingest-time
`chunk_min_chars` coalescing) and the measured trade-offs are their own decision record:
[chunk-sizing-and-retrieval-filter.md](chunk-sizing-and-retrieval-filter.md).

## Interaction with the `RETRIEVAL_MIN_CHUNK_CHARS` filter

The query-time min-chunk-chars filter (in `Retriever`) started as a band-aid for heading-only
chunks polluting top-k. The figure pass adds many short one-line `Figure: …` descriptions and
small tables, so — **measured, not assumed** — the filter's optimum on the re-ingested corpus
turned out to be **higher**, not lower: `RETRIEVAL_MIN_CHUNK_CHARS` default is now **400** (a
transcribed chart *table* is typically well over 400 chars so it survives; a bare caption
falls below and is dropped, which is fine). The earlier note here that it could be "relaxed
toward 0" was **wrong** — see the full sweep and the ingest-time `chunk_min_chars` coalescing
alternative in [chunk-sizing-and-retrieval-filter.md](chunk-sizing-and-retrieval-filter.md).

## Configuration

| Setting | Default | Meaning |
|---|---|---|
| `PDF_HYBRID_RECOVERY` | `1` (on) | Append the text-layer prose Docling dropped into picture regions (chart footnotes/annotations/callouts) via pypdf. Free, lossless, complements `FIGURE_VLM`. Set `0` for a Docling-only parse. |
| `FIGURE_VLM` | `off` | `off` \| `anthropic`. `anthropic` runs the Claude pass. |
| `FIGURE_VLM_MODEL` | `claude-sonnet-5` | Vision model for transcription. |
| `FIGURE_VLM_MIN_PIXELS` | `200` | Skip figures whose long edge is under this (logos/icons/rules). |
| `FIGURE_VLM_MAX_FIGURES` | `0` | Cap figures transcribed per doc (0 = no cap); for cost-bounded test runs. |

Keep `DOCLING_CHART_EXTRACTION` and `DOCLING_PICTURE_DESCRIPTION` **off** when
`FIGURE_VLM=anthropic` — the Claude pass replaces them, and the local stages don't run in
this environment.

## Cost

~1K–1.5K input tokens per figure image + a few hundred output. Rough per-figure cost:
Sonnet 5 ~$0.008, Opus 4.8 ~$0.013, Haiku 4.5 ~$0.003. At ~8.8 figures/doc over ~150
reports (~1,300 figures) a full re-ingest is **~$10 on Sonnet 5** (**~$5 with the Batch
API**, 50% off, offline). Default is **Sonnet 5** — strong chart transcription at ~40% of
Opus cost; move to Opus 4.8 for the hardest/densest charts, or Haiku 4.5 to cut cost
further if quality holds on your figures.

## Incremental (sync) vs. full re-ingest (batch)

There is no runtime auto-detection of "one new report" vs. "the whole corpus" — the same
`IngestionService.run_once` per-file hash/skip loop drives both, and it discovers how many
files changed only as it walks them. So the sync-vs-batch choice is **keyed off the entry
point**, not inferred:

| Entry point | Nature | Figure transcription |
|---|---|---|
| Background scheduler (`ingest/scheduler.py`) → `run_once` → the loader's inline pass | a few figures, want them live now | **synchronous** `messages.create` (`figure_vlm.py`) |
| `scripts/ingest_bulk_batch.py` (manual bulk load / reload) | many figures, offline, cost matters | **Batch API** (`figure_batch.py`), 50% cheaper |

Both share the *same* prompt, image encoding, request shape (`figure_vlm.figure_user_content`),
and figure selection (`figure_vlm.iter_figure_slots`) — only the transport differs.

### The batch bulk loader: backfill vs. incremental

`scripts/ingest_bulk_batch.py` does the **whole** ingest (parse → chunk → embed → store →
manifest), pooling every figure into one Batch submission. PDFs get their figures transcribed;
non-PDF files (`.txt`/`.docx`) have no figures and are ingested plainly, so one run covers a
mixed tree. It has two intents, because the manifest records a file's *content hash* and *that*
it was ingested — never *whether it had figures*:

- **`--reprocess-all` (backfill / full reload)** — process **every** file regardless of the
  manifest. This is the *only* way to add figures to an already-ingested corpus: enabling
  figures doesn't change a PDF's bytes, so its hash still matches and incremental mode would
  skip it. Per document it does **delete-then-reindex** — the doc's chunks are rebuilt from the
  figure-injected Markdown (figures land inline, in the right section), not patched. Re-embedding
  the text is free on the local embedder, and the rebuild also lands the latest chunk-quality
  fixes (`chunk_markdown` / `split_sections`).
- **default (incremental)** — process only **new or changed** files (content hash vs. the
  manifest). The recurring bulk load of newly-arrived reports; do *not* use it to backfill.

Run with the same `RAG_PROVIDER` / `VECTOR_BACKEND` / `DATABASE_URL` as the server:

    # first time you enable figures on the existing corpus — backfill everything:
    python scripts/ingest_bulk_batch.py <folder> --reprocess-all

    # thereafter, a cheap recurring load of just the new reports:
    python scripts/ingest_bulk_batch.py <folder>

### Restart / crash recovery

The batch job is **checkpointed into four phases** under `--work-dir` (default `.figure_batch`,
which must persist between runs). Re-running the exact same command resumes from the persisted
phase — it never restarts from scratch:

1. **collect** — docling-parse each PDF, save its page Markdown + figure crops (one JSON per
   doc). A restart skips docs already collected, so the expensive re-parse is never redone.
   A single PDF failing to parse is recorded and skipped, never aborting the collect.
2. **submit** — one batch for all figures. The `batch_id` is persisted **the instant the API
   returns**, before anything else, so a crash right after submission re-attaches to the
   in-flight batch instead of paying to resubmit it. (If the process dies in the microsecond
   before the id is written, the re-run logs a warning and resubmits — check `batches.list()`.)
3. **poll** — the batch runs *server-side*; a crash here loses nothing, the re-run reads the
   persisted `batch_id` and resumes polling. This is the whole reason to use Batch: the
   hours-long wait survives the client dying.
4. **write** — splice each result into its page and ingest (delete-then-reindex, then stamp
   the `FileState` manifest). A crash mid-write resumes at the first unwritten doc; stamping
   the manifest also tells the live scheduler the doc is current so it won't re-ingest it.

Per-request failures inside the batch (an errored/expired figure) lose *that figure*, never
the document — the same isolation the synchronous pass gives. If a corpus ever exceeds the
100k-request batch cap, split it and run the job per subtree.

## If we ever want docling's native structured chart2csv

The only route is fixing the local vision stack — pinned transformers 4.x in an isolated
ingest venv so Granite Vision runs — then setting `DOCLING_CHART_EXTRACTION=1`. For RAG,
Option B already gives the practical result (the numbers as retrievable text), so this is
optional.
