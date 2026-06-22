"""Engineer-only debug routes: inspect the index and test retrieval.

Hidden from the public schema (``include_in_schema=False``) and gated behind a
debug key (``require_debug_key``); disabled entirely unless DEBUG_API_KEY is set.
Not part of the frontend-facing API.
"""

from typing import Annotated

from fastapi import APIRouter, Depends, Query
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from industryiq.api.deps import get_pipeline
from industryiq.api.security import require_debug_key
from industryiq.core.pipeline import RagPipeline

Pipeline = Annotated[RagPipeline, Depends(get_pipeline)]

router = APIRouter(tags=["debug"], include_in_schema=False)


class Chunk(BaseModel):
    id: str
    text: str
    source: str | None = None


class ChunksResponse(BaseModel):
    count: int
    chunks: list[Chunk]


class ScoredChunk(BaseModel):
    id: str
    score: float
    text: str
    source: str | None = None


class RetrieveResponse(BaseModel):
    query: str
    count: int
    chunks: list[ScoredChunk]


@router.get("/debug/chunks", dependencies=[Depends(require_debug_key)])
def list_chunks(pipeline: Pipeline, limit: int = 100) -> ChunksResponse:
    items = pipeline.list_chunks(limit=limit)
    chunks = [
        Chunk(id=chunk_id, text=metadata.get("text", ""), source=metadata.get("source"))
        for chunk_id, metadata in items
    ]
    return ChunksResponse(count=len(chunks), chunks=chunks)


@router.get("/debug/retrieve", dependencies=[Depends(require_debug_key)])
def debug_retrieve(
    pipeline: Pipeline, q: Annotated[str, Query(min_length=1)], k: int = 5
) -> RetrieveResponse:
    """Return what retrieval surfaces for query ``q`` (ranked, with scores)."""
    hits = pipeline.retrieve(q, k=k)
    chunks = [
        ScoredChunk(
            id=hit.id,
            score=hit.score,
            text=hit.metadata.get("text", ""),
            source=hit.metadata.get("source"),
        )
        for hit in hits
    ]
    return RetrieveResponse(query=q, count=len(chunks), chunks=chunks)


_DEBUG_UI_HTML = """<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>industryiq - debug</title>
  <style>
    body { font-family: system-ui, sans-serif; margin: 2rem; }
    input, button { padding: .5rem; font-size: 1rem; }
    p { margin: .5rem 0; }
    table { border-collapse: collapse; margin-top: 1rem; width: 100%; }
    th, td { border: 1px solid #ccc; padding: .5rem; text-align: left; vertical-align: top; }
    th { background: #f3f3f3; }
    td.score { text-align: right; font-variant-numeric: tabular-nums; white-space: nowrap; }
    .err { color: #b00; }
  </style>
</head>
<body>
  <h1>Index &amp; retrieval debug</h1>
  <p><input id="key" type="password" placeholder="debug key" size="30"></p>
  <p>
    <input id="query" type="text" placeholder="query to test retrieval" size="40">
    <button onclick="retrieve()">Retrieve</button>
    <button onclick="loadAll()">Load all chunks</button>
  </p>
  <p id="status"></p>
  <table id="tbl">
    <thead><tr><th>#</th><th>score</th><th>source</th><th>text</th></tr></thead>
    <tbody></tbody>
  </table>
  <script>
    async function call(url, hasScore) {
      const key = document.getElementById('key').value;
      const status = document.getElementById('status');
      const tbody = document.querySelector('#tbl tbody');
      tbody.innerHTML = '';
      status.textContent = 'Loading...';
      try {
        const res = await fetch(url, { headers: { 'X-Debug-Key': key } });
        if (!res.ok) {
          status.className = 'err';
          status.textContent = 'Error ' + res.status + ' - check your key';
          return;
        }
        const data = await res.json();
        status.className = ''; status.textContent = data.count + ' chunk(s)';
        data.chunks.forEach((c, i) => {
          const tr = document.createElement('tr');
          const cells = [
            { v: String(i + 1) },
            { v: hasScore ? Number(c.score).toFixed(3) : '', cls: 'score' },
            { v: c.source || '' },
            { v: c.text },
          ];
          cells.forEach(({ v, cls }) => {
            const td = document.createElement('td');
            td.textContent = v;
            if (cls) td.className = cls;
            tr.appendChild(td);
          });
          tbody.appendChild(tr);
        });
      } catch (e) { status.className = 'err'; status.textContent = String(e); }
    }
    function retrieve() {
      const q = document.getElementById('query').value;
      if (!q) { document.getElementById('status').textContent = 'Enter a query first'; return; }
      call('/debug/retrieve?q=' + encodeURIComponent(q) + '&k=5', true);
    }
    function loadAll() { call('/debug/chunks', false); }
  </script>
</body>
</html>"""


@router.get("/debug-ui", response_class=HTMLResponse)
def debug_ui() -> str:
    return _DEBUG_UI_HTML
