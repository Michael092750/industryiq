"""FastAPI application: thin HTTP layer over :class:`RagPipeline`.

Routes validate input, call the pipeline, and serialize the result. They contain
no RAG logic of their own -- that all lives in ``ragproject.core``.
"""

import os
import tempfile
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, FastAPI, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from ragproject.api.chat_routes import router as chat_router
from ragproject.api.deps import get_pipeline
from ragproject.api.security import require_debug_key
from ragproject.config import get_settings
from ragproject.core.loaders import SUPPORTED_EXTENSIONS, load
from ragproject.core.pipeline import RagPipeline

Pipeline = Annotated[RagPipeline, Depends(get_pipeline)]


class IngestRequest(BaseModel):
    text: str = Field(min_length=1)
    source: str | None = None


class IngestResponse(BaseModel):
    chunk_ids: list[str]


class QueryRequest(BaseModel):
    question: str = Field(min_length=1)
    k: int = Field(default=5, ge=1)


class Source(BaseModel):
    text: str
    score: float


class QueryResponse(BaseModel):
    answer: str
    sources: list[Source]


class Chunk(BaseModel):
    id: str
    text: str
    source: str | None = None


class ChunksResponse(BaseModel):
    count: int
    chunks: list[Chunk]


app = FastAPI(title="ragproject")

# Allow the browser frontend (a different origin) to call the API.
# Explicit origins from config, plus any localhost port for local dev.
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(get_settings().cors_origins),
    allow_origin_regex=r"http://localhost:\d+",
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/ingest")
def ingest(request: IngestRequest, pipeline: Pipeline) -> IngestResponse:
    chunk_ids = pipeline.ingest_text(request.text, source=request.source)
    return IngestResponse(chunk_ids=chunk_ids)


@app.post("/ingest/file")
def ingest_file(file: UploadFile, pipeline: Pipeline) -> IngestResponse:
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


# Engineer-only endpoints: hidden from the public schema (`include_in_schema=False`)
# and gated behind a debug key (`require_debug_key`). Disabled unless DEBUG_API_KEY is set.
debug_router = APIRouter(
    prefix="/debug",
    tags=["debug"],
    include_in_schema=False,
    dependencies=[Depends(require_debug_key)],
)


@debug_router.get("/chunks")
def list_chunks(pipeline: Pipeline, limit: int = 100) -> ChunksResponse:
    items = pipeline.list_chunks(limit=limit)
    chunks = [
        Chunk(id=chunk_id, text=metadata.get("text", ""), source=metadata.get("source"))
        for chunk_id, metadata in items
    ]
    return ChunksResponse(count=len(chunks), chunks=chunks)


_DEBUG_UI_HTML = """<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>ragproject - debug</title>
  <style>
    body { font-family: system-ui, sans-serif; margin: 2rem; }
    input, button { padding: .5rem; font-size: 1rem; }
    table { border-collapse: collapse; margin-top: 1rem; width: 100%; }
    th, td { border: 1px solid #ccc; padding: .5rem; text-align: left; vertical-align: top; }
    th { background: #f3f3f3; }
    .err { color: #b00; }
  </style>
</head>
<body>
  <h1>Indexed chunks</h1>
  <input id="key" type="password" placeholder="debug key" size="30">
  <button onclick="load()">Load chunks</button>
  <p id="status"></p>
  <table id="tbl">
    <thead><tr><th>#</th><th>source</th><th>text</th></tr></thead>
    <tbody></tbody>
  </table>
  <script>
    async function load() {
      const key = document.getElementById('key').value;
      const status = document.getElementById('status');
      const tbody = document.querySelector('#tbl tbody');
      tbody.innerHTML = '';
      status.textContent = 'Loading...';
      try {
        const res = await fetch('/debug/chunks', { headers: { 'X-Debug-Key': key } });
        if (!res.ok) {
          status.className = 'err';
          status.textContent = 'Error ' + res.status + ' - check your key';
          return;
        }
        const data = await res.json();
        status.className = ''; status.textContent = data.count + ' chunk(s)';
        data.chunks.forEach((c, i) => {
          const tr = document.createElement('tr');
          [String(i + 1), c.source || '', c.text].forEach(v => {
            const td = document.createElement('td'); td.textContent = v; tr.appendChild(td);
          });
          tbody.appendChild(tr);
        });
      } catch (e) { status.className = 'err'; status.textContent = String(e); }
    }
  </script>
</body>
</html>"""


@app.get("/debug-ui", include_in_schema=False, response_class=HTMLResponse)
def debug_ui() -> str:
    return _DEBUG_UI_HTML


@app.post("/query")
def query(request: QueryRequest, pipeline: Pipeline) -> QueryResponse:
    result = pipeline.query(request.question, k=request.k)
    sources = [Source(text=hit.metadata.get("text", ""), score=hit.score) for hit in result.hits]
    return QueryResponse(answer=result.answer, sources=sources)


app.include_router(debug_router)
app.include_router(chat_router)
