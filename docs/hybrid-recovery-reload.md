# Hybrid recovery reload: the batch-path bypass & the OCR-degradation finding

**Status:** resolved (2026-07-07). The corpus was rebuilt with hybrid pypdf recovery in both
stores; all four previously-lost facts are back and the benchmark scores all 50 queries with
zero `expected_missing`.

This records what went wrong while landing [hybrid text recovery](figure-ingestion.md#hybrid-text-recovery-the-completeness-net)
into the corpus, the decisions taken, and the fix. It's the sequel to
[chunk-sizing-and-retrieval-filter.md](chunk-sizing-and-retrieval-filter.md) (which diagnosed
the four Docling-dropped facts) and [figure-ingestion.md](figure-ingestion.md) (the recovery
mechanism itself).

## Issue 1 — the batch-path bypass (silent, caught by verification)

Hybrid recovery was implemented in `loaders._load_pdf_pages_docling` (the synchronous parse),
unit-tested, and validated end-to-end (Docling drops `29.4`; recovery restores it). We then ran
a full `--reprocess-all` reload and, on verifying the fresh corpus, found `29.4` **still
missing**. Two facts that *did* resolve (Stargate, Thinking Machines) turned out to have come
back incidentally via OCR/parser-version differences — **not** the fix.

**Root cause:** the bulk **Batch-API** ingestion has its *own* parallel Docling parse,
`figure_batch._docling_pages_and_doc`, which "mirrors `loaders._load_pdf_pages_docling`" but
called `export_to_markdown` directly and **never applied the hybrid net.** Two independent
parse implementations; fixing one silently missed the other. Every bulk re-ingest runs through
the batch path, so the corpus never got the recovery.

**Lesson:** the two parse paths must stay in lockstep. This is now enforced by a regression
test (below) and a note in figure-ingestion.md.

## Issue 2 — OCR-on amplified the loss far beyond the four benchmark facts

The reload ran `DOCLING_OCR=1` (faithful to how the corpus was originally built). On this
32 GB Windows box, sustained Docling + RapidOCR fragments the native heap until an allocation
fails — **`std::bad_alloc` per page**, and a failed page emits **no Docling text at all**. The
collect logged **~2,000 `bad_alloc` lines across 208 of 210 docs**.

Once hybrid recovery was actually applied, it recovered **+10.3M characters** — because it had
to backfill **whole empty pages** from pypdf, not just figure footnotes. So the Docling content
loss was **corpus-wide**, not limited to the four facts the benchmark happens to probe. The
flip side: for those crashed pages the corpus is now **pypdf-quality** (messier reading order),
since pypdf's flat text is what filled them.

## Decisions

1. **Fix at the source, not just patch the corpus.** Wire `_apply_hybrid_recovery` into
   `figure_batch._docling_pages_and_doc` (respecting `PDF_HYBRID_RECOVERY`), so every future
   bulk ingest applies it. Lock it with a regression test.
2. **Remediate the existing corpus cheaply — migrate + rechunk, don't re-parse.** Rather than a
   full re-parse (hours) + a second paid figure batch, we applied hybrid recovery to the
   already-saved batch **plans** (a per-doc pypdf re-read, cheap) and ran `--rechunk`, which
   **re-fetches the already-paid batch results** and re-chunks/re-writes each doc. No re-parse,
   **no second API charge.** This is sound because a fixed re-parse would hit the *same* OCR
   `bad_alloc` empties and produce the *same* whole-page pypdf backfill — so the migration
   yields an identical outcome for free.
3. **Keep OCR-on for this run, but recommend OCR-off next time.** OCR-on was kept for parity
   with the original corpus, but its instability is what forced the large pypdf backfills.
   See the open recommendation.

## Solution (what shipped)

| Change | File |
|---|---|
| Hybrid recovery wired into the **batch** parse (the bypass fix) | `core/figure_batch.py::_docling_pages_and_doc` |
| Regression test: batch parse must apply hybrid, like the loader | `tests/test_figure_batch.py::test_docling_pages_and_doc_applies_hybrid_recovery` |
| Doc note that **both** parse paths apply the net | `docs/figure-ingestion.md` |
| Corpus rebuilt (migrate plans + `--rechunk`) to pgvector **and** Milvus | `VECTOR_BACKEND=both` |
| `expected_missing` flags cleared for the four recovered facts | `benchmarks/queries.json` |

The migration itself (`_apply_hybrid_recovery` over each saved plan's pages) was a one-off; the
durable fix is the `figure_batch` wiring, so it never needs repeating.

## Verification

- **All four Docling-dropped facts resolve** in the fresh corpus: `29.4%` variable-rate-debt,
  Stargate `$100–500B`, Thinking Machines `$2B seed`, life-insurance `$500B` (also a dropped
  footnote, not a missing document as first assumed).
- **Benchmark scores all 50 queries** (was 46), **zero `expected_missing`**, no hard-fail.
- **Retrieval split (pgvector dense):** `29.4` and Stargate retrieve at **#1**; Thinking Machines
  and life-insurance are **in the corpus but rank >5** — a *retrieval-quality* limit of dense-only
  search on exact numeric tokens (Milvus hybrid + BM25 is expected to surface them; see the
  regression doc's hybrid comparison), **not** content loss.
- **Corpus size:** 28,454 → **38,434 chunks** (+10.3M chars recovered).
- **Cost:** one Haiku 4.5 figure batch (~6,738 figures, ~$8); the fix + rebuild added **no**
  further API charge (results reused via `--rechunk`).

## Open recommendation — re-ingest with OCR off

The +10.3M-char backfill is a symptom: OCR-on crashed Docling so widely that hybrid rescued
whole pages with pypdf's messier text. A future re-ingest with **`DOCLING_OCR=0`** should:

- **crash far less** (the `bad_alloc` storm is OCR-driven), so Docling keeps its clean
  reading-order output on those pages instead of emitting nothing;
- leave hybrid recovering just the **targeted figure footnotes/annotations** (a few KB/doc),
  not whole pages — a smaller, cleaner corpus.

The born-digital reports keep most text in the PDF text layer, which pypdf + non-OCR Docling
both read, so OCR's marginal value (bitmap-only text) is low relative to its instability cost.
A secondary refinement: improve whole-doc-mode recovery attribution — on a page-count mismatch
it currently appends all recovered text to the **last** page (imperfect citations) rather than
per-page.
