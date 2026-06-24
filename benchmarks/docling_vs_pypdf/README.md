# Docling vs pypdf — PDF parsing benchmark

Run 2026-06-24 against real reports from the industry-report library, to decide
whether to back `core/loaders.py::load_pdf_pages` with Docling instead of pypdf.
**CPU-only (no CUDA), OCR off** (no OCR engine installed — see the caveat below).

The deciding question is the Milvus **`section` column**: it is currently *always
empty* because pypdf has no notion of document structure. Docling produces a real
heading hierarchy and structure-aware chunks, so it could populate `section` —
the question is whether that's worth the cost and the failure modes.

**Corpus:** 12 deliberately diverse docs — a 1-page graphic one-pager, consultancy
decks (Accenture), standards/gov text (NIST), table-heavy finance (IMF, FDIC),
multilateral (WEF, WIPO), think-tank/academic (RAND, Brookings, HAI). Both engines
process the **first 10 pages** of each file (apples-to-apples; bounds runtime).

**Reproduce:** `python bench_pdf.py` (defaults to this corpus, 10 pages, OCR off).
Writes `out/results.json` + per-file `out/<stem>.pypdf.txt` / `.docling.md` dumps.
Override with `--root`, `--pages N`, `--ocr`, or explicit file args.

## Summary

| Document | pypdf s | docling s | pypdf chars | docling chars | empty pg (pypdf) | hdrs | tables | pics | chunks w/ section |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| SIA OnePager (1p) | 0.02 | 2.32 | 2,303 | 1,950 | 0 | 3 | 0 | 11 | 2 / 2 |
| WIPO factsheet (136) | 0.17 | 8.35 | 17,084 | 17,907 | 0 | 20 | 1 | 10 | 22 / 22 |
| Accenture (023) | 0.08 | 7.48 | 15,902 | 14,081 | 0 | 16 | 1 | 7 | 19 / 19 |
| Travel (030) | 0.12 | 7.54 | 11,240 | 9,884 | 1 | 8 | 1 | 9 | 11 / 11 |
| NIST AI 600-1 (044) | 0.28 | 8.65 | 23,676 | 25,335 | 0 | 14 | 1 | 3 | 35 / 35 |
| WEF FinServ (051) | 0.18 | 10.16 | 23,426 | 18,741 | 0 | 17 | 2 | 15 | 20 / 21 |
| HAI Index ch1 (006) | 0.19 | 8.74 | 14,043 | 12,146 | 0 | 21 | 2 | 15 | 23 / 23 |
| RAND (110) | 0.11 | 8.58 | 21,484 | 22,141 | 1 | 18 | 2 | 2 | 71 / 71 |
| Brookings (104) | 0.25 | 9.23 | 15,880 | 15,653 | 0 | 9 | 2 | 6 | 18 / 18 |
| WIPO pub (135) | 0.57 | 7.78 | 11,577 | 11,373 | 1 | 19 | 0 | 6 | 24 / 24 |
| IMF GFSR | 3.17 | 21.85 | 14,396 | **25,629** | 1 | 14 | **4** | 4 | 22 / 22 |
| FDIC QBP | 0.41 | 9.44 | 16,077 | 14,280 | 1 | 22 | 0 | **17** | 21 / 22 |

**Totals:** pypdf ≈ 5.5 s, docling ≈ 110 s (**~20× slower**) · section-tagged chunks
**288 / 290 (99%)** · 5 / 12 docs had ≥1 near-empty page in their first 10.

Char and structure counts are deterministic; the `secs` columns are from one
CPU-only run and vary ±20% between runs.

## Finding 1 — Section coverage: 0% → 99% (the reason to adopt Docling)

Every document, Docling tagged ~all chunks with a real heading path
("Chapter 1: Research and Development", "Policy Recommendations", "Executive
Summary"). That is exactly the `section` value the Milvus store currently stores
as `""`. pypdf provides **none** — it has no structure at all. This held across
all 12 doc types and is the single decisive result.

## Finding 2 — Speed: ~20× slower (~1 s/page on CPU)

110 s vs 5.5 s for 12 × 10 pages. Tolerable for the **background**
`IngestionService` (idempotent, content-hash-skipped — a one-time cost per file),
not for interactive upload. A full-library re-ingest goes from seconds to tens of
minutes; a CUDA GPU would cut this substantially.

## Finding 3 — Text volume is a wash, except table-heavy docs

Totals are within ~1% — both engines read the embedded text layer fine, so raw
text recall is **not** a reason to switch for digital PDFs. Two divergences:

* **Table-heavy docs favor Docling.** IMF GFSR: **+78%** (14,396 → 25,629 chars),
  because pypdf flattens tables into scrambled text while Docling reconstructs
  them as clean Markdown grids (e.g. the GFSR contents table renders with aligned
  page numbers in `imf_org__text.docling.md`).
* **Design-heavy docs read ~15–20% lower under Docling** (Accenture, Travel, WEF,
  HAI) — it drops repeated header/footer and decorative text and the axis-label
  noise (`0% 20% 40% …`, `700 600 500 …`) that pypdf dumps inline. That noise
  currently pollutes embeddings, so *fewer* chars here is arguably *better*.

## Finding 4 — Without OCR, both engines lose chart/image-rendered data

This run was text-layer-only (no OCR engine installed). The `pictures` and
`empty_pages` columns are the tell: image-heavy docs (**FDIC 17 pics, SIA 11, WEF
15, HAI 15**) bake text and even whole tables into images. The sharpest example:

> **FDIC QBP detected 0 tables** despite being a banking-statistics report — its
> tables are rendered as images, so Docling-without-OCR misses them *just like
> pypdf does*.

Values that live only inside charts (per-country percentages, bar-chart numbers)
are dropped by Docling (it classifies the chart as `<!-- image -->`); pypdf keeps
them but as unreadable noise. **Closing this gap needs `--ocr` (plus an OCR engine
like `rapidocr-onnxruntime`), which this benchmark did not measure**, and would
add significant time. Chart-only numbers would need a separate chart→text (VLM)
step regardless.

## Conclusion

Docling is a real upgrade for **structure** — it populates `section` on ~99% of
chunks and gives coherent, de-noised chunks (better embeddings/retrieval),
especially on table-heavy financial reports — at a ~20× CPU-time cost on the
background ingestion path. It is **not** a chart-data extractor for this corpus,
and for scanned / image-rendered-table docs neither engine works until OCR is on.

**Decision:** make Docling the default for the **offline ingestion path** behind a
flag (`PDF_PARSER=pypdf|docling`, default pypdf until verified), with pypdf as the
automatic fallback on failure — but first run `--ocr` on the image-heavy subset
(SIA, FDIC, WEF) to learn whether enabling OCR turns this from a section-only win
into a section + scanned-recovery win.

`out/` holds the raw dumps and `results.json` — artifacts, delete anytime.
