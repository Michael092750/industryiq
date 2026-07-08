"""Bulk-load a report tree end to end through the Anthropic Batch API (checkpointed).

The batch counterpart to ``ingest_bulk.py``: it does the *whole* ingest -- parse -> chunk ->
embed -> store -> manifest -- but pools every figure's vision call into one Batch submission
(50% cheaper than live calls), and it is crash-resumable across all four phases (collect ->
submit -> poll -> write) via a durable ``--work-dir``. PDFs get their figures transcribed;
non-PDF files (``.txt``/``.docx``) have no figures and are ingested plainly, so one run
covers a mixed tree.

Two intents, chosen by ``--reprocess-all``:

* **default (incremental)** -- process only *new or changed* files (content hash vs. the
  manifest). This is the recurring bulk load of newly-arrived reports.
* **``--reprocess-all``** -- process *every* file regardless of the manifest. Use this to
  **backfill figures** into an already-ingested corpus: enabling figures does not change a
  PDF's bytes, so its manifest hash still matches and incremental mode would skip it -- only
  a full reprocess adds figures to files that were ingested without them. It re-ingests text
  too (delete-then-reindex), which is free on the local embedder and also lands the latest
  chunk-quality fixes.

Run with the SAME env as ``ingest_bulk`` (RAG_PROVIDER / VECTOR_BACKEND / DATABASE_URL) so
the ingest-time embedder + store match the live app:

    # first time you enable figures on the existing corpus -- backfill everything:
    python scripts/ingest_bulk_batch.py <folder> --reprocess-all

    # thereafter, a cheap recurring load of just the new reports:
    python scripts/ingest_bulk_batch.py <folder>

``--work-dir`` (default ``.figure_batch``) holds the durable checkpoint and MUST persist
between runs for the resume to work. Delete it to start a clean job.
"""

import argparse
import logging
import os
import sys
from pathlib import Path

# generate_picture_images is only turned on when FIGURE_VLM != off; the collector calls
# docling directly (not the inline pass), so this just enables the figure crops.
os.environ.setdefault("FIGURE_VLM", "anthropic")

from industryiq.core import figure_batch  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="folder tree of reports (as in ingest_bulk)")
    parser.add_argument(
        "--reprocess-all",
        action="store_true",
        help="process every file regardless of the manifest (figure backfill / full reload); "
        "default is incremental -- only new or changed files",
    )
    parser.add_argument(
        "--collect-only",
        action="store_true",
        help="parse + save figures but STOP before the paid batch submit, so the exact figure "
        "count/cost can be inspected; re-run without this flag to resume into submit",
    )
    parser.add_argument(
        "--rechunk",
        action="store_true",
        help="re-ingest a finished job from its saved plans with the current chunker/config "
        "(re-fetches batch results, re-chunks + re-embeds + re-writes; no re-parse/re-transcribe)",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=Path(".figure_batch"),
        help="durable checkpoint directory (must persist across restarts)",
    )
    parser.add_argument(
        "--poll-seconds", type=float, default=30.0, help="how often to poll the batch status"
    )
    args = parser.parse_args()
    if not args.root.is_dir():
        sys.exit(f"not a folder: {args.root}")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.rechunk:
        rewound = figure_batch.reset_for_rechunk(args.work_dir)
        logging.info("rechunk: rewound %d written plans to the write phase", rewound)
    mode = "reprocess-all (backfill)" if args.reprocess_all else "incremental (new/changed only)"
    logging.info("bulk batch load: %s, mode=%s", args.root, mode)
    job = figure_batch.run(
        args.root,
        args.work_dir,
        poll_seconds=args.poll_seconds,
        incremental=not args.reprocess_all,
        collect_only=args.collect_only,
    )
    print(f"\nbulk batch job finished in phase: {job.phase}")


if __name__ == "__main__":
    main()
