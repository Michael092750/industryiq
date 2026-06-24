"""Admin routes: populate the shared knowledge base.

``POST /admin/ingest`` is key-gated (``require_admin_key`` / ``X-Admin-Key``) --
it's how an admin loads documents into the shared index that chat retrieves
from. End users never call it; they add their own files per-conversation via the
chat document upload. ``GET /admin/ui`` serves a tiny upload page (public -- it
just collects the key client-side, exactly like the debug page).
"""

import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, UploadFile
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


class IngestResponse(BaseModel):
    chunk_ids: list[str]


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


@router.post("/ingest", dependencies=[Depends(require_admin_key)])
def ingest_file(file: UploadFile, pipeline: Pipeline) -> IngestResponse:
    """Ingest an uploaded file into the shared knowledge base."""
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type {suffix!r}; supported: {sorted(SUPPORTED_EXTENSIONS)}",
        )
    # Write the upload to a temp file so the path-based loaders can read it.
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file.file.read())
        tmp_path = tmp.name
    try:
        # source is the original upload name (the temp path is meaningless to the user).
        chunk_ids = pipeline.ingest_file(tmp_path, source=file.filename)
    finally:
        os.unlink(tmp_path)
    return IngestResponse(chunk_ids=chunk_ids)


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
  </style>
</head>
<body>
  <h1>Ingest into the shared knowledge base</h1>
  <p><input id="key" type="password" placeholder="admin key" size="40"></p>
  <p>
    <input id="file" type="file" accept=".pdf,.docx,.txt">
    <button onclick="ingest()">Ingest</button>
  </p>
  <div id="drop" class="drop">Drop a .pdf, .docx, or .txt file here</div>
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
    drop.addEventListener('click', () => fileInput.click());
    drop.addEventListener('dragover', (e) => { e.preventDefault(); drop.classList.add('over'); });
    drop.addEventListener('dragleave', () => drop.classList.remove('over'));
    drop.addEventListener('drop', (e) => {
      e.preventDefault(); drop.classList.remove('over');
      if (e.dataTransfer.files.length) { fileInput.files = e.dataTransfer.files; ingest(); }
    });
    async function ingest() {
      const key = document.getElementById('key').value;
      const status = document.getElementById('status');
      if (!fileInput.files.length) {
        status.className = ''; status.textContent = 'Choose a file first'; return;
      }
      const name = fileInput.files[0].name;
      const form = new FormData();
      form.append('file', fileInput.files[0]);
      status.className = ''; status.textContent = 'Ingesting ' + name + '...';
      try {
        const res = await fetch('/admin/ingest', {
          method: 'POST', headers: { 'X-Admin-Key': key }, body: form,
        });
        if (!res.ok) {
          status.className = 'err';
          const hint = res.status === 401 ? ' - check your admin key'
                     : res.status === 404 ? ' - admin endpoint disabled (no key configured)'
                     : res.status === 415 ? ' - unsupported file type' : '';
          status.textContent = 'Error ' + res.status + hint;
          return;
        }
        const data = await res.json();
        status.className = 'ok';
        status.textContent = 'Ingested "' + name + '" - ' + data.chunk_ids.length + ' chunk(s).';
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
