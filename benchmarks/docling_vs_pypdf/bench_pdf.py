"""Benchmark: pypdf (the current loader) vs Docling on real report PDFs.

Decides whether to back ``core/loaders.py::load_pdf_pages`` with Docling. The
deciding question is the Milvus ``section`` column -- it is currently always
empty because pypdf has no notion of document structure -- weighed against
Docling's cost (speed) and its failure modes (chart/scanned data dropped when OCR
is off).

Both engines process only the first ``--pages`` pages of each file, so the
comparison is apples-to-apples and the run stays bounded on large reports. Per
file, per engine, it records:

* pypdf   -- ``secs``, ``chars``, ``empty_pages`` (a near-empty page is a scanned
  page whose text lives in an image; the signal that OCR would help).
* docling -- ``secs``, ``chars`` (from the exported Markdown), ``section_headers``,
  ``tables``, ``pictures``, ``chunks`` and ``chunks_with_section`` (how many of
  Docling's structure-aware chunks carry a heading path -- i.e. could populate
  the ``section`` column).

Outputs ``<out>/results.json`` plus per-file text dumps (``<stem>.pypdf.txt`` and
``<stem>.docling.md``) for eyeballing the actual extractions, and prints a summary
table. The dumps and JSON are artifacts -- delete ``<out>`` anytime.

OCR note: ``--ocr`` enables Docling's OCR pipeline, which needs an OCR engine
installed (e.g. ``pip install rapidocr-onnxruntime``). Without it the run is
text-layer-only and image-rendered text/tables are not recovered by either engine.

Usage:
    python bench_pdf.py                          # default corpus, first 10 pages, OCR off
    python bench_pdf.py --pages 15 --ocr
    python bench_pdf.py --root D:/pdfs AI/a.pdf Finance/b.pdf
"""

import argparse
import json
import time
from pathlib import Path
from typing import Any

# The report library that feeds the RAG project (see the project's report-library
# note). Override with --root; the corpus paths below are relative to it.
DEFAULT_ROOT = Path(r"C:\Users\micha\OneDrive\Study YZ\LLM Camp\insdustry_report_pdf")

# A deliberately diverse 12-doc corpus: a 1-page graphic one-pager, consultancy
# decks (Accenture), standards/gov text (NIST), table-heavy finance (IMF, FDIC),
# multilateral (WEF, WIPO), and think-tank/academic (RAND, Brookings, HAI).
CORPUS = [
    r"Semiconductor\2026_SIA_StandardIndustry_OnePager_02c48.pdf",
    r"AI\136_docs-en-pdf-generative-ai-factsheet.pdf",
    r"AI\023_Accenture-A-New-Era-of-Generative-AI-for-Everyone.pdf",
    r"AI\030_The-Travel-Industrys-New-Trip-Final.pdf",
    r"AI\044_NIST.AI.600-1.pdf",
    r"AI\051_WEF_Artificial_Intelligence_in_Financial_Services_2025.pdf",
    r"AI\006_hai_ai-index-report-2025_chapter1_final.pdf",
    r"AI\110_RAND_RRA3888-1.pdf",
    r"AI\104_2019.11.20_brookingsmetro_what-jobs-are-affected-by-ai_report_muro-whiton-maxim.pdf",
    r"AI\135_wipo_pub_1055.pdf",
    r"Finance\imf_org__text.pdf",
    r"Finance\fdic_gov__quarterly-banking-profile-fourth-quarter-2025.pdf",
]

# A page yielding fewer characters than this is treated as "empty" -- almost
# always a full-page chart/scan whose text pypdf cannot reach.
EMPTY_PAGE_CHARS = 50


def run_pypdf(path: Path, cap: int, out: Path) -> dict[str, Any]:
    """Current loader path: pypdf text extraction over the first ``cap`` pages."""
    import pypdf

    t0 = time.perf_counter()
    pages = pypdf.PdfReader(str(path)).pages[:cap]
    texts = [(p.extract_text() or "") for p in pages]
    secs = round(time.perf_counter() - t0, 2)

    (out / f"{path.stem}.pypdf.txt").write_text("\n\n".join(texts), encoding="utf-8")
    return {
        "secs": secs,
        "pages": len(texts),
        "chars": sum(len(t) for t in texts),
        "empty_pages": sum(1 for t in texts if len(t.strip()) < EMPTY_PAGE_CHARS),
    }


def build_converter(ocr: bool):
    """A Docling converter with table structure on and OCR per ``ocr``."""
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption

    opts = PdfPipelineOptions(do_ocr=ocr, do_table_structure=True)
    return DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)}
    )


def run_docling(path: Path, cap: int, converter, out: Path) -> dict[str, Any]:
    """Docling path: structured parse over the first ``cap`` pages.

    Also runs the HybridChunker so we can report how many chunks carry a heading
    path -- the metric that decides whether ``section`` can be populated.
    """
    from docling.chunking import HybridChunker
    from docling_core.types.doc.labels import DocItemLabel

    t0 = time.perf_counter()
    doc = converter.convert(str(path), page_range=(1, cap)).document
    secs = round(time.perf_counter() - t0, 2)

    md = doc.export_to_markdown()
    (out / f"{path.stem}.docling.md").write_text(md, encoding="utf-8")

    chunks = list(HybridChunker().chunk(doc))
    sections = [" > ".join(h) for ch in chunks if (h := getattr(ch.meta, "headings", None))]
    return {
        "secs": secs,
        "chars": len(md),
        "section_headers": sum(1 for it in doc.texts if it.label == DocItemLabel.SECTION_HEADER),
        "tables": len(doc.tables),
        "pictures": len(doc.pictures),
        "chunks": len(chunks),
        "chunks_with_section": len(sections),
        "sample_sections": sections[:3],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="pypdf vs Docling PDF-parsing benchmark")
    parser.add_argument(
        "files", nargs="*", help="PDFs relative to --root (default: built-in corpus)"
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="report-library root")
    parser.add_argument(
        "--pages", type=int, default=10, help="pages per file to process (default 10)"
    )
    parser.add_argument(
        "--ocr", action="store_true", help="enable Docling OCR (needs an OCR engine)"
    )
    parser.add_argument(
        "--out", type=Path, default=Path(__file__).parent / "out", help="output dir"
    )
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    rels = args.files or CORPUS
    converter = build_converter(args.ocr)

    rows: list[dict[str, Any]] = []
    for rel in rels:
        path = args.root / rel
        name = Path(rel).name
        print(f"\n=== {name} ===", flush=True)
        rec: dict[str, Any] = {"file": name}
        for engine in ("pypdf", "docling"):
            try:
                rec[engine] = (
                    run_pypdf(path, args.pages, args.out)
                    if engine == "pypdf"
                    else run_docling(path, args.pages, converter, args.out)
                )
                shown = {k: v for k, v in rec[engine].items() if k != "sample_sections"}
                print(f"  {engine:<8}{shown}", flush=True)
            except Exception as exc:  # noqa: BLE001 -- one bad file mustn't abort the batch
                rec[engine] = {"err": f"{type(exc).__name__}: {exc}"}
                print(f"  {engine:<8}FAIL {rec[engine]['err']}", flush=True)
        rows.append(rec)
        (args.out / "results.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")

    ok = [r for r in rows if "secs" in r.get("pypdf", {}) and "secs" in r.get("docling", {})]
    pp_t = sum(r["pypdf"]["secs"] for r in ok)
    dl_t = sum(r["docling"]["secs"] for r in ok)
    ch = sum(r["docling"]["chunks"] for r in ok)
    sec = sum(r["docling"]["chunks_with_section"] for r in ok)
    print(
        f"\n{len(ok)}/{len(rows)} ok | pypdf {pp_t:.1f}s vs docling {dl_t:.1f}s "
        f"({dl_t / pp_t:.0f}x) | section coverage {sec}/{ch} chunks",
        flush=True,
    )
    print(f"-> {args.out / 'results.json'}", flush=True)


if __name__ == "__main__":
    main()
