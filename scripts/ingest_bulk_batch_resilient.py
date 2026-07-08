"""Crash-resilient supervisor for ``ingest_bulk_batch`` (for OCR-heavy corpora).

The batch bulk loader runs docling *in-process* during its collect phase. With OCR on,
sustained docling+RapidOCR runs can fragment the native heap until an allocation fails and
the process SIGSEGVs -- fatal to a single long run. This wrapper runs the loader in a
*subprocess* and, when it dies, relaunches a **fresh** process that resumes from the batch
job's checkpoint (collected docs are skipped; a submitted batch is re-attached, not
resubmitted; written docs are skipped). A clean heap each time sidesteps the fragmentation.

If one document crashes the parser even in a fresh process (no progress made), it is marked
``failed`` in the work dir so the next attempt moves past it -- and, importantly, the write
phase then leaves that document's *existing* chunks untouched rather than wiping them.

Run with the SAME env as ``ingest_bulk_batch`` (RAG_PROVIDER / VECTOR_BACKEND / DATABASE_URL,
and DOCLING_OCR=1 for the OCR path this wrapper exists to survive):

    DOCLING_OCR=1 VECTOR_BACKEND=both \
      python scripts/ingest_bulk_batch_resilient.py <folder> --reprocess-all --work-dir DIR
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("FIGURE_VLM", "anthropic")

from industryiq.core import figure_batch  # noqa: E402
from industryiq.core.loaders import SUPPORTED_EXTENSIONS  # noqa: E402

# Safety cap so a pathological corpus can't loop forever.
_MAX_ATTEMPTS = 120


def _supported_files(root: Path) -> list[Path]:
    exts = {s.lower() for s in SUPPORTED_EXTENSIONS}
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in exts)


def _resume_target(root: Path, work: Path, files: list[Path]) -> Path | None:
    """The first file with no plan yet -- where the next collect attempt will resume."""
    for p in files:
        source = p.relative_to(root).as_posix()
        if figure_batch.load_doc_plan(work, source) is None:
            return p
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--reprocess-all", action="store_true")
    parser.add_argument("--collect-only", action="store_true")
    parser.add_argument("--work-dir", type=Path, default=Path(".figure_batch"))
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    args = parser.parse_args()
    if not args.root.is_dir():
        sys.exit(f"not a folder: {args.root}")

    files = _supported_files(args.root)
    if not files:
        sys.exit(f"no ingestable files under {args.root}")

    child = [
        sys.executable,
        "-u",
        "scripts/ingest_bulk_batch.py",
        str(args.root),
        "--work-dir",
        str(args.work_dir),
        "--poll-seconds",
        str(args.poll_seconds),
    ]
    if args.reprocess_all:
        child.append("--reprocess-all")
    if args.collect_only:
        child.append("--collect-only")

    for attempt in range(1, _MAX_ATTEMPTS + 1):
        target = _resume_target(args.root, args.work_dir, files)
        job = figure_batch.load_job(args.work_dir)
        phase = job.phase if job else "collecting"
        tgt = target.relative_to(args.root).as_posix() if target else "none (collect complete)"
        print(f"\n=== attempt {attempt}: phase={phase}, resume target={tgt} ===", flush=True)
        proc = subprocess.run(child)
        if proc.returncode == 0:
            print(f"\nDone: batch bulk load finished cleanly after {attempt} attempt(s).")
            return

        # Non-zero exit. If we were still collecting and the resume target didn't advance,
        # that document crashed a fresh process -> mark it failed so we move past it.
        after = _resume_target(args.root, args.work_dir, files)
        if phase == "collecting" and target is not None and after == target:
            src = target.relative_to(args.root).as_posix()
            figure_batch.mark_document_failed(args.work_dir, target, src)
            print(
                f"  !! {src} crashed the parser in a fresh process (exit {proc.returncode}); "
                f"marked failed (existing chunks kept), advancing.",
                flush=True,
            )
        else:
            print(f"  .. subprocess exited {proc.returncode} after progress; resuming.", flush=True)

    still = _resume_target(args.root, args.work_dir, files)
    print(f"\nStopped after {_MAX_ATTEMPTS} attempts; still pending from {still}.")


if __name__ == "__main__":
    main()
