"""Checkpointed, restartable figure transcription for a full corpus re-ingest.

The synchronous figure pass (:mod:`industryiq.core.figure_vlm`, wired into the loader)
transcribes each figure with a live ``messages.create`` call -- right for the scheduler's
incremental, one-report-at-a-time ingest, where you want the figures live immediately. For
a **full corpus re-ingest** (~1,300 figures) that path is neither cheap nor crash-safe: it
pays standard pricing and, if the process dies after hour two, has nothing durable to
resume from.

This module runs the same transcription through the **Anthropic Batch API** (50% cheaper,
async) as a job broken into four checkpointed phases, so a crash *anywhere* resumes instead
of restarting:

1. **collect** -- docling-parse each PDF (no live VLM call), save its page Markdown and its
   figure crops to a work directory. One JSON per document; a restart skips documents
   already collected. Re-parsing the corpus is the expensive part, so it is never redone.
2. **submit** -- build one Batch request per saved figure and submit it. The returned
   ``batch_id`` is persisted **the instant it comes back**, before anything else -- so a
   restart re-attaches to the in-flight batch instead of submitting (and paying for) a
   second one.
3. **poll** -- the batch runs *server-side*; this just waits. A crash here loses nothing:
   a restart reads the persisted ``batch_id`` and resumes polling. This is the whole point
   of the Batch API -- the hours-long wait survives the client dying.
4. **write** -- pull the results, splice each transcription into its page Markdown, and
   ingest the document (delete-then-reindex, then stamp the manifest). Reuses the same
   ``FileState`` manifest as :class:`industryiq.core.ingestion.IngestionService`, so a
   crash mid-write resumes at the first unwritten document.

Per-request failures inside the batch (an errored or expired figure) lose *that figure*,
never the document -- exactly like the synchronous pass isolates a failed call.

See ``docs/figure-ingestion.md`` (the "Batch re-ingest" section) for the decision record.
"""

import hashlib
import json
import logging
import time
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from industryiq.config import Settings, get_settings
from industryiq.core.figure_vlm import (
    FIGURE_MAX_TOKENS,
    figure_user_content,
    inject_figures,
    iter_figure_slots,
)
from industryiq.core.ingestion.manifest import ManifestCache, manifest_metadata
from industryiq.core.ingestion.models import FileState
from industryiq.core.loaders import SUPPORTED_EXTENSIONS

logger = logging.getLogger(__name__)

# The batch job's phases, in order. ``run`` advances through them and each is independently
# resumable, so re-running after a crash picks up wherever the persisted phase left off.
PHASES = ("collecting", "submitting", "submitted", "writing", "done")

# Anthropic caps a single batch at 256 MB (and 100k requests). Our corpus of figure images
# far exceeds 256 MB, so submit packs figures into chunks under this byte budget (kept a
# margin below 256 MB for the JSON request wrappers) -- one batch per chunk.
_MAX_BATCH_BYTES = 200 * 1024 * 1024
_HASH_BLOCK = 1 << 20


# --------------------------------------------------------------------------- #
# Value types (persisted as JSON in the work directory)                        #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class FigureSlot:
    """One picture's slot in reading order.

    ``custom_id`` is the Batch request id when the figure was selected for transcription,
    or ``None`` for a slot left empty (too small, unreadable, or past the cap). ``image``
    is the crop's path relative to the work dir, present only for a selected figure.
    """

    page_no: int
    custom_id: str | None = None
    image: str | None = None


@dataclass
class DocPlan:
    """Everything needed to (re)ingest one document without re-parsing its PDF.

    Written once at collect time; ``written`` flips to ``True`` after the document's chunks
    land, so a restart in the write phase skips it.
    """

    source: str
    content_hash: str
    size: int
    title: str
    metadata: dict[str, Any]
    pages: list[str]
    slots: list[FigureSlot] = field(default_factory=list)
    written: bool = False
    # Set by the resilient supervisor when a document repeatedly crashes the parser (e.g. an
    # OCR SIGSEGV). The write phase then skips it entirely -- crucially it does NOT
    # delete-then-reindex, so the document keeps whatever chunks it already had instead of
    # being wiped by a failed re-parse.
    failed: bool = False

    def to_json(self) -> dict[str, Any]:
        data = asdict(self)
        return data

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "DocPlan":
        slots = [FigureSlot(**s) for s in data.get("slots", [])]
        return cls(
            source=data["source"],
            content_hash=data["content_hash"],
            size=data["size"],
            title=data["title"],
            metadata=data.get("metadata", {}),
            pages=data["pages"],
            slots=slots,
            written=data.get("written", False),
            failed=data.get("failed", False),
        )


@dataclass(frozen=True)
class JobState:
    """The batch job's durable cursor: which phase it is in and the in-flight batch ids.

    ``batch_ids`` is a list because the corpus's figures exceed the Batch API's 256 MB
    per-batch cap and are split across several batches; the count also drives resumable
    submit (chunks already submitted are skipped on a re-run).
    """

    root: str
    model: str
    phase: str = "collecting"
    batch_ids: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Work-directory layout + (de)serialization                                    #
# --------------------------------------------------------------------------- #


def _doc_key(source: str) -> str:
    """A short, filesystem-safe, collision-free key for a document ``source``."""
    return hashlib.sha256(source.encode("utf-8")).hexdigest()[:12]


def _custom_id(doc_key: str, index: int) -> str:
    """A Batch ``custom_id`` for the ``index``-th figure of a document (unique per batch)."""
    return f"{doc_key}-{index:04d}"


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(_HASH_BLOCK), b""):
            digest.update(block)
    return digest.hexdigest()


def _job_path(work: Path) -> Path:
    return work / "job.json"


def _plans_dir(work: Path) -> Path:
    return work / "docs"


def _doc_plan_path(work: Path, source: str) -> Path:
    return _plans_dir(work) / f"{_doc_key(source)}.json"


def _write_json_atomic(path: Path, data: dict[str, Any]) -> None:
    """Write ``data`` as JSON via a temp file + rename, so a crash never leaves a half file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def load_job(work: Path) -> JobState | None:
    """The persisted job cursor, or ``None`` if this work dir has no job yet."""
    path = _job_path(work)
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    # Migrate a legacy single ``batch_id`` (pre multi-batch) into ``batch_ids``.
    batch_ids = data.get("batch_ids")
    if batch_ids is None:
        legacy = data.get("batch_id")
        batch_ids = [legacy] if legacy else []
    return JobState(
        root=data["root"],
        model=data["model"],
        phase=data.get("phase", "collecting"),
        batch_ids=batch_ids,
    )


def save_job(work: Path, job: JobState) -> None:
    _write_json_atomic(_job_path(work), asdict(job))


def save_doc_plan(work: Path, plan: DocPlan) -> None:
    _write_json_atomic(_doc_plan_path(work, plan.source), plan.to_json())


def load_doc_plan(work: Path, source: str) -> DocPlan | None:
    path = _doc_plan_path(work, source)
    if not path.is_file():
        return None
    return DocPlan.from_json(json.loads(path.read_text(encoding="utf-8")))


def iter_doc_plans(work: Path) -> Iterator[DocPlan]:
    """Every collected document plan, in a stable (source-sorted) order."""
    plans_dir = _plans_dir(work)
    if not plans_dir.is_dir():
        return
    for path in sorted(plans_dir.glob("*.json")):
        yield DocPlan.from_json(json.loads(path.read_text(encoding="utf-8")))


def mark_document_failed(work: Path, file_path: Path, source: str) -> None:
    """Persist a ``failed`` plan for a document that crashes the parser (e.g. an OCR SIGSEGV).

    Used by the resilient supervisor: after a document has crashed a *fresh* process, this
    records it so collection advances past it and the write phase leaves its existing chunks
    intact (see :class:`DocPlan.failed`). The plan carries no pages or figures.
    """
    save_doc_plan(
        work,
        DocPlan(
            source=source,
            content_hash=_file_hash(file_path),
            size=file_path.stat().st_size,
            title=Path(source).stem,
            metadata={},
            pages=[],
            slots=[],
            failed=True,
        ),
    )


# --------------------------------------------------------------------------- #
# Pure result-assembly (unit-tested offline)                                   #
# --------------------------------------------------------------------------- #


def plan_to_pages(plan: DocPlan, results: dict[str, str]) -> list[str]:
    """Splice batch ``results`` (``custom_id -> markdown``) into a plan's page Markdown.

    Rebuilds the per-page, reading-order figure lists from the plan's slots -- an empty
    string for an unselected slot or a figure whose request is missing/errored -- and
    injects them positionally, exactly as the synchronous pass does. Pure and offline.
    """
    figures_by_page: dict[int, list[str]] = {}
    for slot in plan.slots:
        text = results.get(slot.custom_id, "") if slot.custom_id else ""
        figures_by_page.setdefault(slot.page_no, []).append(text)
    return [
        inject_figures(page_md, figures_by_page.get(n, []))
        for n, page_md in enumerate(plan.pages, start=1)
    ]


def _supported_files(root: Path) -> list[Path]:
    """Every ingestable file under ``root``, in the order the ingest service walks them."""
    return sorted(
        p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )


# --------------------------------------------------------------------------- #
# Phase 1: collect (docling parse + save crops)                                #
# --------------------------------------------------------------------------- #


def _docling_pages_and_doc(path: Path) -> tuple[list[str], Any]:
    """Per-page Markdown (placeholders intact) and the ``DoclingDocument`` for its figures.

    Mirrors ``loaders._load_pdf_pages_docling`` but *without* the inline VLM pass: we want
    the raw ``<!-- image -->`` placeholders here and transcribe them via the batch instead.
    Needs ``generate_picture_images=True`` on the converter (set when ``FIGURE_VLM`` is not
    ``off``), so run the batch job with ``FIGURE_VLM=anthropic``.

    Applies the same ``PDF_HYBRID_RECOVERY`` net as the synchronous loader: it appends the
    text-layer prose Docling drops into picture regions (chart footnotes/annotations) from
    pypdf. Runs before figure injection here (the batch splices transcriptions at write
    time), which is fine -- recovery only appends dropped text and never touches the
    ``<!-- image -->`` placeholders the write phase relies on.
    """
    from industryiq.config import get_settings
    from industryiq.core.loaders import _apply_hybrid_recovery, _get_docling_converter

    doc = _get_docling_converter().convert(str(path)).document
    page_count = len(doc.pages)
    if page_count == 0:
        pages = [doc.export_to_markdown()]
    else:
        pages = [doc.export_to_markdown(page_no=n) for n in range(1, page_count + 1)]
    if get_settings().pdf_hybrid_recovery:
        pages = _apply_hybrid_recovery(path, pages)
    return pages, doc


def collect_document(
    path: Path,
    *,
    source: str,
    metadata: dict[str, Any],
    title: str,
    work: Path,
    settings: Settings,
) -> DocPlan:
    """Parse one PDF, save its figure crops, and return its :class:`DocPlan`.

    Figure selection (min-pixels, cap, empty slots) goes through the *same*
    :func:`iter_figure_slots` the synchronous pass uses, so batch and sync agree on which
    figures are transcribed. Crops for selected figures are written under ``figs/<key>/``.

    Non-PDF documents (``.txt``, ``.docx``) have no figures to batch, so their text is
    captured the normal (synchronous) way and the plan carries no figure slots -- it still
    flows through submit (contributes nothing) and write (ingested plainly) unchanged.
    """
    if path.suffix.lower() != ".pdf":
        from industryiq.core.loaders import load_pages

        return DocPlan(
            source=source,
            content_hash=_file_hash(path),
            size=path.stat().st_size,
            title=title,
            metadata=metadata,
            pages=load_pages(path),
            slots=[],
        )

    pages, doc = _docling_pages_and_doc(path)
    doc_key = _doc_key(source)
    figs_dir = work / "figs" / doc_key
    slots: list[FigureSlot] = []
    for index, (page_no, image) in enumerate(
        iter_figure_slots(
            doc,
            min_pixels=settings.figure_vlm_min_pixels,
            max_figures=settings.figure_vlm_max_figures,
        )
    ):
        if image is None:
            slots.append(FigureSlot(page_no=page_no))  # empty slot, no request
            continue
        figs_dir.mkdir(parents=True, exist_ok=True)
        rel = f"figs/{doc_key}/{index:04d}.png"
        image.convert("RGB").save(work / rel, format="PNG")
        slots.append(FigureSlot(page_no=page_no, custom_id=_custom_id(doc_key, index), image=rel))

    return DocPlan(
        source=source,
        content_hash=_file_hash(path),
        size=path.stat().st_size,
        title=title,
        metadata=metadata,
        pages=pages,
        slots=slots,
    )


def collect_all(
    root: Path,
    work: Path,
    settings: Settings,
    *,
    store: Any = None,
) -> tuple[int, list[tuple[str, str]]]:
    """Collect every not-yet-collected document under ``root``. Returns ``(done, failures)``.

    Idempotent: a document whose plan JSON already exists is skipped, so re-running after a
    crash resumes collection. One document failing to parse is recorded and skipped -- it
    never aborts the collect (mirroring the ingest service's per-file isolation).

    ``store`` selects the two intents this job serves:

    * ``store=None`` (**process-all**) -- collect *every* file regardless of the manifest.
      This is figure **backfill**: the manifest records the PDF's content hash and *that* it
      was ingested, never *whether it had figures*, so enabling figures leaves the hash
      unchanged. Only reprocessing everything can add figures to already-ingested files.
    * ``store`` given (**incremental**) -- skip a file whose content hash already matches its
      manifest entry. This is the recurring **bulk load** of newly-arrived reports: touch
      only the delta. (Do *not* use this to backfill figures -- unchanged files are skipped.)
    """
    from industryiq.core.loaders import load_title

    files = _supported_files(root)
    manifests: ManifestCache = {}
    collected = 0
    failures: list[tuple[str, str]] = []
    for file_path in files:
        rel = file_path.relative_to(root)
        source = rel.as_posix()
        if load_doc_plan(work, source) is not None:
            continue  # already collected in a previous run
        if store is not None:
            prior = store.get_file_state(source)
            if prior is not None and prior.content_hash == _file_hash(file_path):
                continue  # incremental: already ingested and unchanged -> skip
        category = rel.parts[0] if len(rel.parts) > 1 else "uncategorized"
        try:
            metadata: dict[str, Any] = {
                **manifest_metadata(file_path, manifests),
                "category": category,
            }
            title = load_title(file_path) or Path(source).stem
            plan = collect_document(
                file_path,
                source=source,
                metadata=metadata,
                title=title,
                work=work,
                settings=settings,
            )
            save_doc_plan(work, plan)
            collected += 1
            n_figs = sum(1 for s in plan.slots if s.custom_id)
            logger.info("collected %s (%d figures)", source, n_figs)
        except Exception as exc:  # noqa: BLE001 -- one bad PDF mustn't abort the collect
            failures.append((source, f"{type(exc).__name__}: {exc}"))
            logger.warning("collect failed for %s (%s); skipping.", source, exc)
    return collected, failures


# --------------------------------------------------------------------------- #
# Phase 2: submit                                                              #
# --------------------------------------------------------------------------- #


# Anthropic downsamples any image to a ~1568px long edge internally, so sending anything
# larger just wastes batch payload (and can breach the per-image size cap). Downscale to
# this on our side: same tokens/quality the model sees, far smaller upload.
_MAX_IMAGE_EDGE = 1568


def _encode_crop_b64(path: Path) -> str:
    """Base64-PNG a crop, downscaling its long edge to ``_MAX_IMAGE_EDGE`` if larger."""
    import base64
    import io

    from PIL import Image

    with Image.open(path) as im:
        if max(im.size) <= _MAX_IMAGE_EDGE:
            return base64.standard_b64encode(path.read_bytes()).decode("ascii")
        scale = _MAX_IMAGE_EDGE / max(im.size)
        resized = im.convert("RGB").resize((round(im.width * scale), round(im.height * scale)))
        buffer = io.BytesIO()
        resized.save(buffer, format="PNG")
        return base64.standard_b64encode(buffer.getvalue()).decode("ascii")


def iter_encoded_figures(work: Path, *, min_pixels: int = 0) -> Iterator[dict[str, str]]:
    """Yield ``{custom_id, png_b64}`` per saved figure crop, lazily (one at a time).

    ``min_pixels`` (> 0) applies a **submit-time** size filter: a saved crop whose long edge
    is under it is skipped, so its figure gets no batch result and is dropped at write -- how
    a higher threshold is applied *without re-collecting* (crops were saved at collect's lower
    ``FIGURE_VLM_MIN_PIXELS``, and small decorative pictures can be pruned here). Streaming
    keeps only one encoded image in memory at a time, since the whole corpus far exceeds RAM.
    """
    from PIL import Image

    for plan in iter_doc_plans(work):
        for slot in plan.slots:
            if not slot.custom_id or not slot.image:
                continue
            path = work / slot.image
            try:
                if min_pixels > 0:
                    with Image.open(path) as im:
                        if max(im.size) < min_pixels:
                            continue
                yield {"custom_id": slot.custom_id, "png_b64": _encode_crop_b64(path)}
            except Exception:  # noqa: BLE001 -- an unreadable crop is skipped, not fatal
                continue


def build_requests(work: Path, *, min_pixels: int = 0) -> list[dict[str, str]]:
    """Eager form of :func:`iter_encoded_figures` (used in tests)."""
    return list(iter_encoded_figures(work, min_pixels=min_pixels))


def _batch_request(figure: dict[str, str], model: str) -> dict[str, Any]:
    """Wrap one ``{custom_id, png_b64}`` into a Message Batches request."""
    return {
        "custom_id": figure["custom_id"],
        "params": {
            "model": model,
            "max_tokens": FIGURE_MAX_TOKENS,
            "messages": [{"role": "user", "content": figure_user_content(figure["png_b64"])}],
        },
    }


def submit_batch(
    work: Path,
    job: JobState,
    client: Any,
    *,
    min_pixels: int = 0,
    max_batch_bytes: int = _MAX_BATCH_BYTES,
) -> JobState:
    """Submit all figures across one or more batches (the corpus exceeds the 256 MB cap).

    Figures are streamed in a stable order and packed into size-capped chunks; each chunk is
    one batch. Every ``batch_id`` is persisted the instant its ``create`` returns, so a crash
    is safe: on a re-run the chunks already submitted (``len(job.batch_ids)`` of them) are
    skipped and only the rest are sent -- no double-submit, no double-charge. Returns the job
    at ``submitted`` (or ``writing`` if there were no figures at all). ``min_pixels`` prunes
    small crops at submit time (see :func:`iter_encoded_figures`).
    """
    already = len(job.batch_ids)
    batch_ids = list(job.batch_ids)
    chunk_index = 0
    chunk: list[dict[str, Any]] = []
    chunk_bytes = 0
    n_figures = 0

    def flush() -> None:
        nonlocal chunk_index, chunk, chunk_bytes, batch_ids
        if not chunk:
            return
        if chunk_index >= already:  # not yet submitted on a prior run
            logger.info("submitting batch chunk %d (%d figures)...", chunk_index, len(chunk))
            batch = client.messages.batches.create(requests=chunk)
            batch_ids.append(batch.id)
            save_job(work, replace(job, phase="submitting", batch_ids=batch_ids))  # persist FIRST
            logger.info("  batch %d submitted: %s", chunk_index, batch.id)
        chunk_index += 1
        chunk = []
        chunk_bytes = 0

    for figure in iter_encoded_figures(work, min_pixels=min_pixels):
        n_figures += 1
        request = _batch_request(figure, job.model)
        approx = len(figure["png_b64"]) + 512  # image dominates; +overhead for the JSON wrapper
        if chunk and chunk_bytes + approx > max_batch_bytes:
            flush()
        chunk.append(request)
        chunk_bytes += approx
    flush()

    if not batch_ids:
        logger.info("no figures to transcribe; skipping submit.")
        return replace(job, phase="writing", batch_ids=[])
    logger.info("submitted %d figures across %d batch(es).", n_figures, len(batch_ids))
    return replace(job, phase="submitted", batch_ids=batch_ids)


# --------------------------------------------------------------------------- #
# Phase 3: poll                                                                #
# --------------------------------------------------------------------------- #


def poll_until_ended(client: Any, batch_ids: list[str], *, poll_seconds: float = 30.0) -> None:
    """Block until *every* batch has ended. Stateless -- safe to re-enter.

    The batches process on Anthropic's side, so this loop holds no state a crash could lose:
    re-running reads ``batch_ids`` from the job file and resumes polling the unfinished ones.
    """
    remaining = list(batch_ids)
    while remaining:
        still: list[str] = []
        for batch_id in remaining:
            batch = client.messages.batches.retrieve(batch_id)
            if batch.processing_status != "ended":
                still.append(batch_id)
            logger.info(
                "batch %s: %s %s",
                batch_id,
                batch.processing_status,
                getattr(batch, "request_counts", None),
            )
        if not still:
            return
        remaining = still
        time.sleep(poll_seconds)


def fetch_results(client: Any, batch_ids: list[str]) -> dict[str, str]:
    """Pull every ended batch's results into one ``{custom_id: markdown}`` map.

    A per-request failure (errored / expired / canceled) maps to an empty string -- that
    one figure is lost, its document intact -- matching the synchronous pass's isolation.
    """
    results: dict[str, str] = {}
    for batch_id in batch_ids:
        for entry in client.messages.batches.results(batch_id):
            text = ""
            result = entry.result
            if result.type == "succeeded":
                text = "".join(
                    block.text for block in result.message.content if block.type == "text"
                ).strip()
            else:
                logger.warning("figure %s: batch result %s (dropped)", entry.custom_id, result.type)
            results[entry.custom_id] = text
    return results


# --------------------------------------------------------------------------- #
# Phase 4: write                                                               #
# --------------------------------------------------------------------------- #


def write_all(work: Path, results: dict[str, str], pipeline: Any, store: Any) -> int:
    """Ingest every not-yet-written document, splicing in its figure transcriptions.

    Delete-then-reindex per document (so a re-run after a partial write can't duplicate
    chunks), then stamp the ``FileState`` manifest -- which both makes this phase resumable
    and tells the live scheduler the document is current, so it won't re-ingest it. Returns
    the number of documents written.
    """
    written = 0
    for plan in iter_doc_plans(work):
        if plan.written:
            continue
        if plan.failed:
            # A doc that crashed the parser: leave its existing chunks untouched (do NOT
            # delete-then-reindex to nothing) and move on.
            logger.warning(
                "skipping %s (marked failed at collect); existing chunks kept", plan.source
            )
            continue
        pages = plan_to_pages(plan, results)
        pipeline.delete_source(plan.source)
        ids = pipeline.ingest_pages(
            pages, source=plan.source, metadata=plan.metadata, title=plan.title
        )
        store.upsert_file_state(
            FileState(
                source=plan.source,
                size=plan.size,
                content_hash=plan.content_hash,
                chunk_count=len(ids),
                ingested_at=datetime.now(UTC),
            )
        )
        plan.written = True
        save_doc_plan(work, plan)  # checkpoint: this doc won't be redone on restart
        written += 1
        logger.info("wrote %s (%d chunks)", plan.source, len(ids))
    return written


# --------------------------------------------------------------------------- #
# Orchestration                                                                #
# --------------------------------------------------------------------------- #


def reset_for_rechunk(work: Path) -> int:
    """Rewind a finished job to the write phase so it re-ingests from the saved plans.

    Clears every plan's ``written`` flag (leaving ``failed`` plans untouched) and sets the
    job phase back to ``writing``. A subsequent :func:`run` then re-fetches the batch results
    (available ~29 days) and re-chunks + re-embeds + re-writes each document with the *current*
    chunker/config -- no PDF re-parse and no figure re-transcription. Returns the number of
    plans rewound. Use it to re-apply a chunking change to the whole corpus cheaply.
    """
    job = load_job(work)
    if job is not None and job.phase != "writing":
        save_job(work, replace(job, phase="writing"))
    rewound = 0
    for plan in iter_doc_plans(work):
        if plan.written and not plan.failed:
            plan.written = False
            save_doc_plan(work, plan)
            rewound += 1
    return rewound


def _anthropic_client(settings: Settings) -> Any:
    import anthropic

    if not settings.anthropic_api_key:
        raise ValueError("the batch figure job requires ANTHROPIC_API_KEY to be set")
    return anthropic.Anthropic(api_key=settings.anthropic_api_key)


def run(
    root: Path,
    work: Path,
    *,
    poll_seconds: float = 30.0,
    incremental: bool = False,
    collect_only: bool = False,
) -> JobState:
    """Drive the batch figure re-ingest to completion, resuming from the persisted phase.

    Safe to call repeatedly: each call reads the job's phase and does only the work still
    outstanding. The pipeline is built lazily (only when the write phase is reached) so
    collect and submit don't load the embedder or open the vector store.

    ``incremental`` picks the two intents (see :func:`collect_all`): ``False`` (default) is a
    **backfill / full reprocess** -- every file under ``root`` is processed regardless of the
    manifest, which is the only way to add figures to already-ingested files. ``True`` is a
    **recurring bulk load** -- only new or changed files (by content hash vs. the manifest)
    are processed. The manifest store is built once and shared by collect (for the skip) and
    write (for the stamp).

    ``collect_only`` stops after the (free, no-API) collect phase, leaving the job at
    ``submitting`` -- so you can inspect the exact figure count/cost before committing to the
    paid batch. Re-running *without* the flag then resumes straight into submit.
    """
    work.mkdir(parents=True, exist_ok=True)
    settings = get_settings()
    job = load_job(work) or JobState(root=str(root), model=settings.figure_vlm_model)
    save_job(work, job)

    store = None
    if incremental:
        from industryiq.api.deps import _build_ingest_state_store

        store = _build_ingest_state_store(settings)

    if job.phase == "collecting":
        collected, failures = collect_all(root, work, settings, store=store)
        logger.info("collect done: %d newly collected, %d failed", collected, len(failures))
        job = replace(job, phase="submitting")
        save_job(work, job)

    if collect_only:
        logger.info("collect-only: stopping before submit (phase=%s)", job.phase)
        return job

    if job.phase == "submitting":
        # Model + min_pixels are submit-time choices: honor the current env, which may differ
        # from collect (e.g. collect ran as Sonnet at min_pixels 200; submit as Haiku at 300).
        if settings.figure_vlm_model != job.model:
            logger.info("model override: %s -> %s", job.model, settings.figure_vlm_model)
            job = replace(job, model=settings.figure_vlm_model)
            save_job(work, job)
        logger.info(
            "submitting batch; if this process dies after the API accepts it but before the "
            "id is recorded, re-running submits a second batch -- check `batches.list()` if unsure."
        )
        job = submit_batch(
            work, job, _anthropic_client(settings), min_pixels=settings.figure_vlm_min_pixels
        )
        save_job(work, job)

    if job.phase == "submitted":
        if not job.batch_ids:  # no figures anywhere -> submit skipped straight to writing
            job = replace(job, phase="writing")
        else:
            poll_until_ended(_anthropic_client(settings), job.batch_ids, poll_seconds=poll_seconds)
            job = replace(job, phase="writing")
        save_job(work, job)

    if job.phase == "writing":
        from industryiq.api.deps import _build_ingest_state_store, get_pipeline

        results = fetch_results(_anthropic_client(settings), job.batch_ids) if job.batch_ids else {}
        write_store = store if store is not None else _build_ingest_state_store(settings)
        written = write_all(work, results, get_pipeline(), write_store)
        logger.info("write done: %d documents ingested", written)
        job = replace(job, phase="done")
        save_job(work, job)

    return job
