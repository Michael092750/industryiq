# ragproject

A modular, testable Retrieval-Augmented Generation (RAG) system.

## Development

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
pytest
```

## Quality checks

```powershell
ruff check .
mypy src
pytest
```

## Run the API locally

```powershell
python -m uvicorn ragproject.api.app:app --reload
```

Then open <http://127.0.0.1:8000/docs> for the interactive Swagger UI.

Key endpoints (full interactive list at `/docs`):

- `GET /health` — liveness check
- `POST /conversations` — start a chat (`GET /conversations` lists them)
- `POST /conversations/{id}/messages` — ask → `{answer, standalone_question, sources, timings_ms}`
- `POST /conversations/{id}/messages/stream` — same answer, streamed as Server-Sent Events
- `GET /conversations/{id}/messages` — turn history
- `POST /conversations/{id}/documents` — upload a file into that chat session only
- `POST /admin/ingest` — load a file into the shared knowledge base (needs `X-Admin-Key`)
- `GET /debug/retrieve`, `GET /debug/chunks` — inspect retrieval (needs `X-Debug-Key`)

> Note: the default wiring uses offline fakes (`FakeEmbedder`, `FakeLLM`), so
> answers are placeholders. Retrieval and source attribution are fully functional.

## Real answers locally (no AWS)

Set the provider to `anthropic` for real Claude answers plus a local CPU embedder:

```powershell
pip install -e ".[dev,local]"
# in .env:
#   RAG_PROVIDER=anthropic
#   ANTHROPIC_API_KEY=sk-ant-...
```

This uses Claude via the Anthropic API and a local `fastembed` embedder — no AWS.
On deploy, `RAG_PROVIDER=bedrock` (set by `compose.prod.yml`) switches to Amazon
Bedrock automatically, authenticated by the instance IAM role.
<<<<<<< HEAD
=======

## Bulk-ingest reports for local testing

To load many documents at once, organize them so each **top-level subfolder is a
category** (industry), then run [`scripts/ingest_bulk.py`](scripts/ingest_bulk.py):

```text
reports/
  AI/        1.pdf  2.pdf ...
  finance/   1.pdf  2.pdf ...
```

```powershell
# 1) start the shared Postgres store (the script writes straight to it)
docker compose up -d db

# 2) point .env at it, with the same provider the API uses:
#   DATABASE_URL=postgresql://rag:ragpass@localhost:5432/ragproject
#   RAG_PROVIDER=anthropic
#   ANTHROPIC_API_KEY=sk-ant-...

# 3) ingest the whole tree
python scripts/ingest_bulk.py "C:\path\to\reports"
```

The script walks subfolders recursively and stores the top-level folder as a
`category` on every chunk (files directly under the root become `uncategorized`),
so retrieved hits are attributable to an industry. Supported types: `.pdf`,
`.docx`, `.txt`.

> Two gotchas: it needs a **shared store**, so set `DATABASE_URL` (with the
> in-memory store the data lives only inside the API process). And ingest with the
> **same `RAG_PROVIDER`** you query with — a 384-dim local embedder and 1024-dim
> Bedrock Titan are not interchangeable against one store.

## Testing chat + RAG locally

Chat retrieval and ingestion must share a store, so use **Postgres** — with the
in-memory store, chat can't see what you ingested (the chat service and the ingest
pipeline build separate stores in the same process).

1. Start the store and load a knowledge base (as above):

   ```powershell
   docker compose up -d db
   python scripts/ingest_bulk.py "C:\path\to\reports"
   ```

2. Start the API (same `.env`, so it uses the same provider + database):

   ```powershell
   python -m uvicorn ragproject.api.app:app --reload
   ```

3. Drive a conversation from `/docs`, or from PowerShell:

   ```powershell
   $base = "http://127.0.0.1:8000"
   $conv = Invoke-RestMethod -Method Post -Uri "$base/conversations" `
     -ContentType application/json -Body (@{ title = "rag test" } | ConvertTo-Json)
   $body = @{ question = "What are the main risks in the finance reports?" } | ConvertTo-Json
   $resp = Invoke-RestMethod -Method Post -Uri "$base/conversations/$($conv.id)/messages" `
     -ContentType application/json -Body $body
   $resp.answer
   $resp.sources | Format-Table score, document
   ```

`sources` should be non-empty and point at your documents — that confirms RAG
retrieved from the knowledge base rather than the model answering blind. Ask a
follow-up and check `standalone_question` to see history-aware query rewriting.

Faster options: `pytest` runs `ChatService` end to end with offline fakes (no
server, no key); or set `RAG_PROVIDER=fake` for a no-cost plumbing smoke test.
>>>>>>> load-local-file
