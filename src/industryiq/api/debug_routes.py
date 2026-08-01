"""Engineer-only debug routes: inspect the index and test retrieval.

Hidden from the public schema (``include_in_schema=False``) and gated behind a
debug key (``require_debug_key``); disabled entirely unless DEBUG_API_KEY is set.
Not part of the frontend-facing API.
"""

from typing import Annotated, Any

from fastapi import APIRouter, Depends, Query
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from industryiq.api.deps import (
    get_blackboard,
    get_pipeline,
    get_redis,
    get_run_ledger,
    get_task_queue,
)
from industryiq.api.security import require_debug_key
from industryiq.core.agents.worker import DEFAULT_QUEUE
from industryiq.core.pipeline import RagPipeline
from industryiq.core.redis_client import ping as redis_ping

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


class RedisHealth(BaseModel):
    configured: bool  # REDIS_URL is set (a client was built)
    reachable: bool  # the server answered PING


class RunNode(BaseModel):
    node_id: str
    status: str  # "done" | "pending"
    summary: str | None = None


class InFlightView(BaseModel):
    node_id: str | None
    consumer: str
    idle_ms: float
    attempt: int


class RunView(BaseModel):
    run_id: str
    question: str
    nodes: list[RunNode]
    inflight: list[InFlightView]
    dead: list[str | None]
    events: list[dict[str, Any]]  # the run ledger, in append order


class QueueStats(BaseModel):
    queue: str
    pending: int  # outstanding = undelivered + claimed-unacked
    inflight: int  # claimed but not yet acked
    dead: int  # dead-lettered


@router.get("/debug/redis", dependencies=[Depends(require_debug_key)])
def redis_health() -> RedisHealth:
    """Report whether Redis is configured and reachable (onboarding smoke test)."""
    client = get_redis()
    if client is None:
        return RedisHealth(configured=False, reachable=False)
    return RedisHealth(configured=True, reachable=redis_ping(client))


@router.get("/debug/runs/{run_id}", dependencies=[Depends(require_debug_key)])
def inspect_run(run_id: str) -> RunView:
    """Reconstruct one agent run: its plan, per-node status, live in-flight tasks,
    dead-letters, and the full event ledger.

    The single place a distributed run's scattered state is reassembled -- what you
    watch while a worker is killed and a peer resumes the task.
    """
    events = get_run_ledger().events(run_id)
    blackboard = get_blackboard()
    queue = get_task_queue()

    plan_event = next((e for e in events if e.get("event") == "plan_created"), None)
    question = str(plan_event.get("question", "")) if plan_event else ""
    specs = plan_event.get("nodes", []) if plan_event else []

    nodes: list[RunNode] = []
    for spec in specs:
        node_id = str(spec.get("node_id"))
        result = blackboard.read(run_id, node_id)
        summary = result.get("summary") if isinstance(result, dict) else None
        nodes.append(
            RunNode(
                node_id=node_id,
                status="done" if result is not None else "pending",
                summary=summary,
            )
        )

    inflight = [
        InFlightView(
            node_id=flight.task.payload.get("node_id"),
            consumer=flight.consumer,
            idle_ms=flight.idle_ms,
            attempt=flight.delivery_count,
        )
        for flight in queue.inflight(DEFAULT_QUEUE)
        if flight.task.payload.get("run_id") == run_id
    ]
    dead = [
        task.payload.get("node_id")
        for task in queue.dead(DEFAULT_QUEUE)
        if task.payload.get("run_id") == run_id
    ]
    return RunView(
        run_id=run_id, question=question, nodes=nodes, inflight=inflight, dead=dead, events=events
    )


@router.get("/debug/agents/queues", dependencies=[Depends(require_debug_key)])
def queue_stats() -> QueueStats:
    """Live counts for the agents queue: backlog, in-flight, and dead-lettered."""
    queue = get_task_queue()
    return QueueStats(
        queue=DEFAULT_QUEUE,
        pending=queue.pending(DEFAULT_QUEUE),
        inflight=len(queue.inflight(DEFAULT_QUEUE)),
        dead=len(queue.dead(DEFAULT_QUEUE)),
    )


@router.get("/debug/chunks", dependencies=[Depends(require_debug_key)])
def list_chunks(pipeline: Pipeline, limit: int = 100) -> ChunksResponse:
    items = pipeline.list_chunks(limit=limit)
    chunks = [
        Chunk(id=chunk_id, text=metadata.get("text", ""), source=metadata.get("source"))
        for chunk_id, metadata in items
    ]
    return ChunksResponse(count=len(chunks), chunks=chunks)


# Upper bound on how many ranked chunks the debug retrieve returns when the
# caller asks for "all" (``k <= 0``): high enough to cover the whole index in
# practice, capped so a pathological corpus can't stream unbounded rows.
_MAX_DEBUG_RESULTS = 100_000


@router.get("/debug/retrieve", dependencies=[Depends(require_debug_key)])
def debug_retrieve(
    pipeline: Pipeline, q: Annotated[str, Query(min_length=1)], k: int = 0
) -> RetrieveResponse:
    """Return what retrieval surfaces for query ``q`` (ranked, with scores).

    ``k`` caps the number of chunks returned; ``k <= 0`` (the default) ranks and
    returns the entire index.
    """
    hits = pipeline.retrieve(q, k=k if k > 0 else _MAX_DEBUG_RESULTS)
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
      call('/debug/retrieve?q=' + encodeURIComponent(q), true);
    }
    function loadAll() { call('/debug/chunks', false); }
  </script>
</body>
</html>"""


@router.get("/debug-ui", response_class=HTMLResponse)
def debug_ui() -> str:
    return _DEBUG_UI_HTML
