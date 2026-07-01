"""Admin routes: populate the shared knowledge base.

``POST /admin/ingest`` is key-gated (``require_admin_key`` / ``X-Admin-Key``) --
it's how an admin loads documents into the shared index that chat retrieves
from. End users never call it; they add their own files per-conversation via the
chat document upload. ``GET /admin/ui`` serves a tiny upload page (public -- it
just collects the key client-side, exactly like the debug page).
"""

import csv
import io
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from industryiq.api.deps import get_ingestion_service, get_pipeline
from industryiq.api.security import require_admin_key
from industryiq.core.ingestion import IngestionPathError, IngestionService, IngestRunResult
from industryiq.core.loaders import SUPPORTED_EXTENSIONS
from industryiq.core.pipeline import RagPipeline

Pipeline = Annotated[RagPipeline, Depends(get_pipeline)]
Ingestion = Annotated[IngestionService, Depends(get_ingestion_service)]

router = APIRouter(prefix="/admin", tags=["admin"])


class IngestedFile(BaseModel):
    """Per-file result within a batch ingest."""

    filename: str
    chunk_ids: list[str]


class IngestResponse(BaseModel):
    # ``chunk_ids`` is the flat union across every file (kept for backward
    # compatibility); ``files`` breaks the same ids down per uploaded file.
    chunk_ids: list[str]
    files: list[IngestedFile]


class IngestJobConfigBody(BaseModel):
    """Admin-set ingestion schedule: where to scan, how often, and on/off."""

    path: str | None = None
    interval_minutes: int = Field(default=60, ge=1)
    enabled: bool = False


class IngestJobConfigOut(BaseModel):
    path: str | None
    interval_minutes: int
    enabled: bool
    updated_at: datetime | None


class IngestRunOut(BaseModel):
    ingested: int
    updated: int
    skipped: int
    deleted_chunks: int
    chunks_added: int
    by_category: dict[str, int]
    failures: list[tuple[str, str]]
    started_at: datetime | None
    finished_at: datetime | None
    busy: bool


class IngestJobStatusOut(BaseModel):
    config: IngestJobConfigOut
    last_run: IngestRunOut | None


def _run_out(result: IngestRunResult) -> IngestRunOut:
    return IngestRunOut(
        ingested=result.ingested,
        updated=result.updated,
        skipped=result.skipped,
        deleted_chunks=result.deleted_chunks,
        chunks_added=result.chunks_added,
        by_category=result.by_category,
        failures=result.failures,
        started_at=result.started_at,
        finished_at=result.finished_at,
        busy=result.busy,
    )


# Provenance the bulk/scheduled ingest derives from the folder tree + sibling
# manifest.csv; offered here as optional inputs so an admin upload can be tagged
# (and later faceted/filtered) identically. Blank values are omitted, so an
# untagged upload never stores an empty promoted column.
_METADATA_FIELDS = ("category", "publisher", "source_type", "published_date")
# The manifest CSV's recognized columns are exactly the metadata fields; any other
# column is ignored.
_MANIFEST_COLUMNS = frozenset(_METADATA_FIELDS)


def _ingest_upload(
    upload: UploadFile, pipeline: RagPipeline, metadata: dict[str, str] | None
) -> list[str]:
    """Ingest one upload via a temp file (the path-based loaders need a path)."""
    suffix = Path(upload.filename or "").suffix.lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(upload.file.read())
        tmp_path = tmp.name
    try:
        # source is the original upload name (the temp path is meaningless to the user).
        return pipeline.ingest_file(tmp_path, source=upload.filename, metadata=metadata)
    finally:
        os.unlink(tmp_path)


def _parse_manifest(upload: UploadFile) -> list[dict[str, str]]:
    """Parse a manifest CSV into one metadata dict per row, in row order.

    Keeps only the recognized columns and only non-blank cells (unknown columns
    are ignored, blanks left unset). Raises 400 for a manifest that isn't UTF-8
    CSV or whose header has none of the recognized columns -- both signal a
    wrong-format upload the admin should see rather than have silently ignored.
    """
    try:
        text = upload.file.read().decode("utf-8-sig")  # utf-8-sig tolerates an Excel BOM
    except UnicodeDecodeError as exc:
        raise HTTPException(status_code=400, detail="manifest must be a UTF-8 CSV file") from exc
    reader = csv.DictReader(io.StringIO(text))
    headers = {(name or "").strip().lower() for name in reader.fieldnames or []}
    if not headers & _MANIFEST_COLUMNS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"manifest header has none of the recognized columns {sorted(_MANIFEST_COLUMNS)}"
            ),
        )
    return [
        {
            (key or "").strip().lower(): value.strip()
            for key, value in row.items()
            if (key or "").strip().lower() in _MANIFEST_COLUMNS and value and value.strip()
        }
        for row in reader
    ]


@router.post("/ingest", dependencies=[Depends(require_admin_key)])
def ingest_file(
    files: list[UploadFile],
    pipeline: Pipeline,
    category: Annotated[str | None, Form()] = None,
    publisher: Annotated[str | None, Form()] = None,
    source_type: Annotated[str | None, Form()] = None,
    published_date: Annotated[str | None, Form()] = None,
    manifest: UploadFile | None = None,
) -> IngestResponse:
    """Ingest one or more uploaded files into the shared knowledge base.

    Per-file metadata comes from an optional ``manifest`` CSV: one row per file,
    matched to ``files`` *by order* (row 1 -> first file, and so on). Its recognized
    columns are ``category``/``publisher``/``source_type``/``published_date`` (any
    subset; other columns ignored). A blank cell -- or no manifest at all -- falls
    back to the like-named inline form field, which is the batch-wide default. So
    the inline fields alone tag every file the same way, and a manifest overrides
    per file where it wants to differ.

    These fields mirror the provenance the bulk ingest reads from the folder name
    and sibling ``manifest.csv``, so uploads are retrievable/faceted just like
    bulk-ingested files. Blank values are omitted (never stored as empty).
    ``published_date`` is stored verbatim; an ISO-sortable value (e.g. ``2024`` or
    ``2024-03-15``) works best as a range filter, matching the bulk path.

    Validation is up front: an unsupported file type (415) or a manifest whose row
    count doesn't match the number of files (400) rejects the whole batch before
    anything is ingested, so a batch never lands half-applied.
    """
    unsupported = sorted(
        {
            upload.filename or "?"
            for upload in files
            if Path(upload.filename or "").suffix.lower() not in SUPPORTED_EXTENSIONS
        }
    )
    if unsupported:
        raise HTTPException(
            status_code=415,
            detail=(
                f"Unsupported file type(s): {unsupported}; "
                f"supported: {sorted(SUPPORTED_EXTENSIONS)}"
            ),
        )
    supplied = (category, publisher, source_type, published_date)
    defaults = {
        field: value.strip()
        for field, value in zip(_METADATA_FIELDS, supplied, strict=True)
        if value and value.strip()
    }
    if manifest is not None:
        rows = _parse_manifest(manifest)
        if len(rows) != len(files):
            raise HTTPException(
                status_code=400,
                detail=(
                    f"manifest has {len(rows)} row(s) but {len(files)} file(s) were uploaded; "
                    "provide exactly one row per file, in the same order"
                ),
            )
        # Manifest cell wins per field; the inline default fills any blank cell.
        per_file = [{**defaults, **row} for row in rows]
    else:
        per_file = [dict(defaults) for _ in files]
    ingested = [
        IngestedFile(
            filename=upload.filename or "",
            chunk_ids=_ingest_upload(upload, pipeline, meta or None),
        )
        for upload, meta in zip(files, per_file, strict=True)
    ]
    return IngestResponse(
        chunk_ids=[cid for result in ingested for cid in result.chunk_ids],
        files=ingested,
    )


def _config_out(service: IngestionService) -> IngestJobConfigOut:
    config = service.config()
    return IngestJobConfigOut(
        path=config.path,
        interval_minutes=config.interval_minutes,
        enabled=config.enabled,
        updated_at=config.updated_at,
    )


@router.get("/ingest-job", dependencies=[Depends(require_admin_key)])
def get_ingest_job(service: Ingestion) -> IngestJobStatusOut:
    """Return the scheduled-ingestion config and the last run's summary."""
    last = service.last_run()
    return IngestJobStatusOut(
        config=_config_out(service),
        last_run=_run_out(last) if last is not None else None,
    )


@router.put("/ingest-job", dependencies=[Depends(require_admin_key)])
def set_ingest_job(body: IngestJobConfigBody, service: Ingestion) -> IngestJobConfigOut:
    """Set the folder to scan, the interval, and whether the schedule is on.

    Takes effect on the scheduler's next poll. The path is the *server's* view of
    the folder (a mounted volume on the live service); it is not validated here so
    it can be set before the volume is in place -- a run reports the error instead.
    """
    config = service.set_config(
        path=body.path or None,
        interval_minutes=body.interval_minutes,
        enabled=body.enabled,
    )
    return IngestJobConfigOut(
        path=config.path,
        interval_minutes=config.interval_minutes,
        enabled=config.enabled,
        updated_at=config.updated_at,
    )


@router.post("/ingest-job/run-now", dependencies=[Depends(require_admin_key)])
def run_ingest_job_now(service: Ingestion) -> IngestRunOut:
    """Run the bulk ingest immediately (sync -> FastAPI threadpool, won't block).

    Scans the configured path once, ingesting new/changed files. Returns the run
    summary; a ``busy`` result means a run was already in progress.
    """
    try:
        result = service.run_once()
    except IngestionPathError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return _run_out(result)


_ADMIN_UI_HTML = """<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>industryiq - admin</title>
  <style>
    body { font-family: system-ui, sans-serif; margin: 2rem; max-width: 640px; }
    input, button { padding: .5rem; font-size: 1rem; }
    p { margin: .75rem 0; }
    .drop { border: 2px dashed #ccc; border-radius: 8px; padding: 1.5rem; text-align: center;
            color: #666; cursor: pointer; }
    .drop.over { border-color: #3778dd; background: #f3f7ff; }
    .err { color: #b00; }
    .ok { color: #0a7d28; }
    pre { background: #f6f6f6; padding: .6rem .8rem; border-radius: 6px; overflow-x: auto; }
    code { background: #f6f6f6; padding: 0 .25rem; border-radius: 3px; }
    .hint { color: #555; font-size: .9rem; }
    .file-list { padding-left: 1.6rem; margin: .5rem 0; }
    .file-list:empty::after { content: "No files selected yet."; color: #888; }
    .file-list li { margin: .3rem 0; }
    .file-list .fname { font-family: ui-monospace, monospace; }
    .file-list button { padding: .1rem .45rem; font-size: .85rem; margin-left: .3rem; }
  </style>
</head>
<body>
  <h1>Ingest into the shared knowledge base</h1>
  <p><input id="key" type="password" placeholder="admin key" size="40"></p>
  <p>
    <input id="file" type="file" accept=".pdf,.docx,.txt" multiple>
    <button onclick="ingest()">Ingest</button>
    <button type="button" onclick="clearFiles()">Clear</button>
  </p>
  <div id="drop" class="drop">Drop .pdf, .docx, or .txt files here (adds to the list below)</div>
  <p>Files to ingest, <strong>in order</strong> &mdash; row N is matched to manifest row N.
     Use &uarr;/&darr; to reorder, &times; to remove:</p>
  <ol id="file-list" class="file-list"></ol>

  <p>Optional <strong>manifest</strong> (CSV) &mdash; per-file metadata, one row per file
     matched <strong>by order</strong> to the list above (row 1 &rarr; first file, &hellip;).
     Row count must equal the number of files.</p>
  <p><input id="manifest" type="file" accept=".csv"></p>
  <p class="hint">Recognized columns (any subset): <code>category</code>, <code>publisher</code>,
     <code>source_type</code>, <code>published_date</code>. Unknown columns are ignored; a blank
     cell falls back to the matching default below.
     Example <code>manifest.csv</code> for two files:</p>
  <pre>category,publisher,source_type,published_date
AI,mckinsey.com,consultancy,2024
Finance,bcg.com,consultancy,2023</pre>

  <p>Metadata defaults &mdash; used when there is no manifest, or a manifest cell is blank:</p>
  <p>
    <label>Category
      <input id="meta-category" type="text" size="14" placeholder="e.g. AI"></label>
    <label>Publisher
      <input id="meta-publisher" type="text" size="14" placeholder="e.g. mckinsey.com"></label>
  </p>
  <p>
    <label>Source type
      <input id="meta-source-type" type="text" size="14" placeholder="e.g. consultancy"></label>
    <label>Published date
      <input id="meta-published-date" type="text" size="10" placeholder="e.g. 2024"></label>
  </p>
  <p id="status"></p>

  <hr>
  <h2>Scheduled bulk ingestion</h2>
  <p>Scan a server folder on an interval and ingest new/changed files (uses the
     admin key above).</p>
  <p><label>Folder path
     <input id="job-path" type="text" size="36" placeholder="/data/reports"></label></p>
  <p><label>Every <input id="job-interval" type="number" min="1" value="60" size="5">
     minute(s)</label></p>
  <p><label><input id="job-enabled" type="checkbox"> Enabled</label></p>
  <p>
    <button onclick="saveJob()">Save schedule</button>
    <button onclick="runJob()">Run now</button>
    <button onclick="loadJob()">Refresh</button>
  </p>
  <p id="job-status"></p>
  <pre id="job-last"></pre>
  <script>
    const drop = document.getElementById('drop');
    const fileInput = document.getElementById('file');
    const manifestInput = document.getElementById('manifest');

    // The reorderable list is the source of truth for order: a native FileList is
    // read-only and reflects the OS picker's order, not the admin's intent. Files
    // accumulate here (picker + drops) until Ingest, so order can be adjusted first.
    let selectedFiles = [];
    function mkBtn(label, title, fn, disabled) {
      const b = document.createElement('button');
      b.type = 'button'; b.textContent = label; b.title = title;
      b.disabled = !!disabled; b.onclick = fn;
      return b;
    }
    function renderFiles() {
      const ol = document.getElementById('file-list');
      ol.innerHTML = '';
      selectedFiles.forEach((f, i) => {
        const li = document.createElement('li');
        const name = document.createElement('span');
        name.className = 'fname'; name.textContent = f.name;
        li.appendChild(name);
        const last = i === selectedFiles.length - 1;
        li.append(
          mkBtn('↑', 'Move up', () => moveFile(i, -1), i === 0),
          mkBtn('↓', 'Move down', () => moveFile(i, 1), last),
          mkBtn('×', 'Remove', () => removeFile(i))
        );
        ol.appendChild(li);
      });
    }
    function addFiles(list) { for (const f of list) selectedFiles.push(f); renderFiles(); }
    function moveFile(i, d) {
      const j = i + d;
      if (j < 0 || j >= selectedFiles.length) return;
      [selectedFiles[i], selectedFiles[j]] = [selectedFiles[j], selectedFiles[i]];
      renderFiles();
    }
    function removeFile(i) { selectedFiles.splice(i, 1); renderFiles(); }
    function clearFiles() { selectedFiles = []; renderFiles(); }

    // Reset the input value after copying so re-picking the same file still fires change.
    fileInput.addEventListener('change', () => {
      addFiles(fileInput.files); fileInput.value = '';
    });
    drop.addEventListener('click', () => fileInput.click());
    drop.addEventListener('dragover', (e) => { e.preventDefault(); drop.classList.add('over'); });
    drop.addEventListener('dragleave', () => drop.classList.remove('over'));
    drop.addEventListener('drop', (e) => {
      e.preventDefault(); drop.classList.remove('over');
      if (e.dataTransfer.files.length) addFiles(e.dataTransfer.files);
    });

    async function ingest() {
      const key = document.getElementById('key').value;
      const status = document.getElementById('status');
      if (!selectedFiles.length) {
        status.className = ''; status.textContent = 'Add a file first'; return;
      }
      const many = selectedFiles.length !== 1;
      const label = many ? selectedFiles.length + ' files' : '"' + selectedFiles[0].name + '"';
      const form = new FormData();
      for (const f of selectedFiles) form.append('files', f);
      // A manifest (if chosen) carries per-file metadata, matched by order server-side.
      if (manifestInput.files.length) form.append('manifest', manifestInput.files[0]);
      // Default metadata: only send fields the admin actually filled in; blanks stay unset.
      for (const [field, id] of [
        ['category', 'meta-category'], ['publisher', 'meta-publisher'],
        ['source_type', 'meta-source-type'], ['published_date', 'meta-published-date'],
      ]) {
        const v = document.getElementById(id).value.trim();
        if (v) form.append(field, v);
      }
      status.className = ''; status.textContent = 'Ingesting ' + label + '...';
      try {
        const res = await fetch('/admin/ingest', {
          method: 'POST', headers: { 'X-Admin-Key': key }, body: form,
        });
        if (!res.ok) {
          status.className = 'err';
          // Surface the server's detail (415 lists bad files; 400 explains a manifest problem).
          let detail = '';
          try { detail = ' - ' + (await res.json()).detail; } catch (e) {}
          const hint = res.status === 401 ? ' - check your admin key'
                     : res.status === 404 ? ' - admin endpoint disabled (no key configured)'
                     : detail;
          status.textContent = 'Error ' + res.status + hint;
          return;
        }
        const data = await res.json();
        status.className = 'ok';
        status.textContent = 'Ingested ' + label + ' - ' + data.chunk_ids.length
          + ' chunk(s) across ' + data.files.length + ' file(s).';
        clearFiles();  // avoid accidentally re-ingesting the same batch
      } catch (e) { status.className = 'err'; status.textContent = String(e); }
    }

    function adminKey() { return document.getElementById('key').value; }
    function renderLast(run) {
      document.getElementById('job-last').textContent = run ? JSON.stringify(run, null, 2) : '';
    }
    async function loadJob() {
      const s = document.getElementById('job-status');
      s.className = ''; s.textContent = 'Loading...';
      try {
        const res = await fetch('/admin/ingest-job', { headers: { 'X-Admin-Key': adminKey() } });
        if (!res.ok) { s.className = 'err'; s.textContent = 'Error ' + res.status; return; }
        const data = await res.json();
        document.getElementById('job-path').value = data.config.path || '';
        document.getElementById('job-interval').value = data.config.interval_minutes;
        document.getElementById('job-enabled').checked = data.config.enabled;
        s.className = 'ok'; s.textContent = 'Loaded.';
        renderLast(data.last_run);
      } catch (e) { s.className = 'err'; s.textContent = String(e); }
    }
    async function saveJob() {
      const s = document.getElementById('job-status');
      const body = {
        path: document.getElementById('job-path').value || null,
        interval_minutes: parseInt(document.getElementById('job-interval').value, 10),
        enabled: document.getElementById('job-enabled').checked,
      };
      s.className = ''; s.textContent = 'Saving...';
      try {
        const res = await fetch('/admin/ingest-job', {
          method: 'PUT',
          headers: { 'X-Admin-Key': adminKey(), 'Content-Type': 'application/json' },
          body: JSON.stringify(body),
        });
        if (!res.ok) { s.className = 'err'; s.textContent = 'Error ' + res.status; return; }
        s.className = 'ok'; s.textContent = 'Saved.';
      } catch (e) { s.className = 'err'; s.textContent = String(e); }
    }
    async function runJob() {
      const s = document.getElementById('job-status');
      s.className = ''; s.textContent = 'Running...';
      try {
        const res = await fetch('/admin/ingest-job/run-now', {
          method: 'POST', headers: { 'X-Admin-Key': adminKey() },
        });
        if (!res.ok) {
          s.className = 'err';
          let detail = '';
          try { detail = ' - ' + (await res.json()).detail; } catch (e) {}
          s.textContent = 'Error ' + res.status + detail;
          return;
        }
        const data = await res.json();
        s.className = 'ok';
        s.textContent = data.busy ? 'A run is already in progress.'
          : ('Done: ' + data.ingested + ' new, ' + data.updated + ' updated, '
             + data.skipped + ' skipped, ' + data.chunks_added + ' chunks.');
        renderLast(data);
      } catch (e) { s.className = 'err'; s.textContent = String(e); }
    }
  </script>
</body>
</html>"""


@router.get("/ui", include_in_schema=False, response_class=HTMLResponse)
def admin_ui() -> str:
    return _ADMIN_UI_HTML
