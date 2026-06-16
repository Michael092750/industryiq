"""Admin routes: populate the shared knowledge base.

``POST /admin/ingest`` is key-gated (``require_admin_key`` / ``X-Admin-Key``) --
it's how an admin loads documents into the shared index that chat retrieves
from. End users never call it; they add their own files per-conversation via the
chat document upload. ``GET /admin/ui`` serves a tiny upload page (public -- it
just collects the key client-side, exactly like the debug page).
"""

import os
import tempfile
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from ragproject.api.deps import get_pipeline
from ragproject.api.security import require_admin_key
from ragproject.core.loaders import SUPPORTED_EXTENSIONS, load
from ragproject.core.pipeline import RagPipeline

Pipeline = Annotated[RagPipeline, Depends(get_pipeline)]

router = APIRouter(prefix="/admin", tags=["admin"])


class IngestResponse(BaseModel):
    chunk_ids: list[str]


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
        text = load(tmp_path)
    finally:
        os.unlink(tmp_path)
    chunk_ids = pipeline.ingest_text(text, source=file.filename)
    return IngestResponse(chunk_ids=chunk_ids)


_ADMIN_UI_HTML = """<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>ragproject - admin</title>
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
  </script>
</body>
</html>"""


@router.get("/ui", include_in_schema=False, response_class=HTMLResponse)
def admin_ui() -> str:
    return _ADMIN_UI_HTML
