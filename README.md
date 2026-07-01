# IndustryIQ

**An AI research agent for industry analysis** — answers questions from a curated
library of consultancy and association reports, with cited sources. Built as a
modular, testable Retrieval-Augmented Generation (RAG) system.

> **Quick nav:** [run locally](#run-the-api-locally) · [web UI](#web-ui-industryiq) ·
> [test end-to-end](#end-to-end-test-full-stack-api--ui) ·
> [stop local services](#stop-and-reset-local-services).
> Deploying to AWS? See **[DEPLOY.md](DEPLOY.md)**.

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

Two ways, depending on what you're doing:

**Whole stack in Docker — one command.** Brings up Postgres + the API and reads your
`.env` automatically (provider, `ANTHROPIC_API_KEY`, secrets), so there's nothing to
type each run:

```powershell
./scripts/start.ps1        # docker compose up -d --build, then waits for /health
```

For local dev, `docker compose` auto-loads [`docker-compose.override.yml`](docker-compose.override.yml),
which bakes the local `fastembed` embedder into the app image (`EXTRAS=local`) and
injects your `.env` into the container (`env_file`) — so `RAG_PROVIDER=anthropic` works
inside Docker. (The explicit `-f` files used on deploy skip this override, so
production stays lean — see [DEPLOY.md](DEPLOY.md).) Stop with `docker compose down`
(keeps data); the first chat is slow while the embedding model downloads in the container.

**Host venv — fast reload loop.** Run just the API in the foreground (in-memory by
default, or against the Dockerized Postgres if `DATABASE_URL` is set):

```powershell
python -m uvicorn industryiq.api.app:app --reload
```

Then open <http://127.0.0.1:8000/docs> for the interactive Swagger UI.

Key endpoints (full interactive list at `/docs`):

- `GET /health` — liveness check
- `POST /auth/register`, `POST /auth/login` — email + password; both return a bearer token
- `GET /auth/me` — the signed-in account
- `POST /conversations` — start a chat (`GET /conversations` lists your chats)
- `PATCH /conversations/{id}` — rename · `DELETE /conversations/{id}` — delete
- `POST /conversations/{id}/messages` — ask → `{answer, standalone_question, sources, timings_ms}`
- `POST /conversations/{id}/messages/stream` — same answer, streamed as Server-Sent Events
- `GET /conversations/{id}/messages` — turn history
- `POST /conversations/{id}/documents` — upload a file into that chat session only
- `POST /admin/ingest` — load a file into the shared knowledge base (needs `X-Admin-Key`)
- `GET /debug/retrieve`, `GET /debug/chunks` — inspect retrieval (needs `X-Debug-Key`)

All `/conversations/*` routes require a bearer token (`Authorization: Bearer <token>`)
and are scoped to their owner — you only ever see your own chats.

> Note: the default wiring uses offline fakes (`FakeEmbedder`, `FakeLLM`), so
> answers are placeholders. Retrieval and source attribution are fully functional.

## Authentication

Accounts are email + password. Register or log in to receive a JSON Web Token,
then send it as `Authorization: Bearer <token>` on every `/conversations/*`
request; each chat is owned by the account that created it. Passwords are
bcrypt-hashed and tokens are signed (HS256), expiring after 24h.

Set a signing secret in `.env` — the built-in dev default is public, so override
it in any real deployment with a long random value:

```powershell
# generate one:  python -c "import secrets; print(secrets.token_urlsafe(48))"
# then in .env:
#   JWT_SECRET=<the value>
#   JWT_EXPIRY_MINUTES=1440   # optional, default 24h
```

User accounts persist in Postgres when `DATABASE_URL` is set; otherwise they
live in memory and are lost on restart.

## Web UI (IndustryIQ)

A React + Vite + TypeScript single-page app lives in [`frontend/`](frontend/): a
dark, GPT-style chat client with email login, a conversation sidebar
(new/rename/delete), answers streamed with a live per-step timer, retrieved
sources, and drag-and-drop document upload. With the API running:

```powershell
cd frontend
npm install      # first time only
npm run dev      # http://localhost:5173
```

See [`frontend/README.md`](frontend/README.md) for details.

## Real answers locally (no AWS)

Set the provider to `anthropic` for real Claude answers plus a local CPU embedder:

```powershell
pip install -e ".[dev,local]"
# in .env:
#   RAG_PROVIDER=anthropic
#   ANTHROPIC_API_KEY=sk-ant-...
```

This uses Claude via the Anthropic API and a local `fastembed` embedder — no AWS.
The `.[local]` install (fastembed) is only needed for the **host venv** path; running
via [`./scripts/start.ps1`](scripts/start.ps1) installs it in the image automatically
and picks up the same `.env`. On deploy, `RAG_PROVIDER=bedrock` (set by
`compose.prod.yml`) switches to Amazon Bedrock automatically, authenticated by the
instance IAM role.

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
#   DATABASE_URL=postgresql://rag:ragpass@localhost:5432/industryiq
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

## Scheduled bulk ingestion (admin-configurable)

The live service can ingest a folder **on a schedule** that an admin sets at
runtime — no redeploy. A background loop scans a server-side folder every _N_
minutes and ingests new/changed files into the shared knowledge base, using the
same code path as the CLI above (so Milvus is the target when
`VECTOR_BACKEND=milvus`). Because it runs **in-process**, it reuses the app's
embedder + store — no ingest-vs-query dimension mismatch.

**Where the files live.** The job reads a path the _server_ can see. In Docker
that's a mounted volume — `docker-compose.yml` mounts `./reports` to
`/data/reports` (read-only). Drop reports under `./reports` with one subfolder
per category, then point the job at `/data/reports`.

**Configure it** via the key-gated admin API (`X-Admin-Key`, the same
`ADMIN_API_KEY` as `/admin/ingest`), or the panel on `GET /admin/ui`:

```powershell
$admin = @{ "X-Admin-Key" = $env:ADMIN_API_KEY }

# set the folder, interval, and turn it on
Invoke-RestMethod "$base/admin/ingest-job" -Method Put -Headers $admin `
  -ContentType application/json `
  -Body (@{ path = "/data/reports"; interval_minutes = 60; enabled = $true } | ConvertTo-Json)

# inspect config + last run
Invoke-RestMethod "$base/admin/ingest-job" -Headers $admin

# ingest immediately (don't wait for the next tick)
Invoke-RestMethod "$base/admin/ingest-job/run-now" -Method Post -Headers $admin
```

**Idempotent re-runs.** Each file is fingerprinted by a content hash (not mtime,
which is unreliable across OneDrive/containers). On every scan: new files are
ingested, **changed** files have their old chunks deleted and are re-ingested, and
unchanged files are skipped — so re-scanning the same folder never duplicates
content. The manifest lives in Postgres, so dedup survives restarts (without
`DATABASE_URL` it's in-memory and each run re-ingests everything).

> **One scheduler per process.** The container runs a single uvicorn worker, so
> exactly one scheduler runs. Running multiple workers would make each schedule
> independently. Disable the loop with `INGEST_SCHEDULER_ENABLED=0`.

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
   python -m uvicorn industryiq.api.app:app --reload
   ```

3. Drive a conversation from `/docs` (click **Authorize** and paste a token), or
   from PowerShell — the chat routes require a bearer token:

   ```powershell
   $base = "http://127.0.0.1:8000"
   # sign in once, then reuse the token (use /auth/login if already registered)
   $cred = @{ email = "you@example.com"; password = "password123" } | ConvertTo-Json
   $tok  = (Invoke-RestMethod -Method Post -Uri "$base/auth/register" `
     -ContentType application/json -Body $cred).access_token
   $auth = @{ Authorization = "Bearer $tok" }

   $conv = Invoke-RestMethod -Method Post -Uri "$base/conversations" -Headers $auth `
     -ContentType application/json -Body (@{ title = "rag test" } | ConvertTo-Json)
   $body = @{ question = "What are the main risks in the finance reports?" } | ConvertTo-Json
   $resp = Invoke-RestMethod -Method Post -Uri "$base/conversations/$($conv.id)/messages" -Headers $auth `
     -ContentType application/json -Body $body
   $resp.answer
   $resp.sources | Format-Table score, document
   ```

`sources` should be non-empty and point at your documents — that confirms RAG
retrieved from the knowledge base rather than the model answering blind. Ask a
follow-up and check `standalone_question` to see history-aware query rewriting.

Faster options: `pytest` runs `ChatService` end to end with offline fakes (no
server, no key); or set `RAG_PROVIDER=fake` for a no-cost plumbing smoke test.

## End-to-end test (full stack: API + UI)

Three layers, fastest first:

| Layer | Command | What it proves |
| --- | --- | --- |
| Backend unit + API | `pytest` | Routing, auth, and chat orchestration end to end with offline fakes — no server, no keys, no database. |
| Backend integration | `pytest -m integration` | The DB-backed stores against real services (needs `DATABASE_URL`; some tests need the Docker services up). |
| Frontend | `cd frontend ; npm run build` | Type-checks (`tsc`) and bundles the UI. |

For a true full-stack run-through, start all three services, then exercise the app:

```powershell
# 1) Postgres (persistent users + chats)
docker compose up -d db

# 2) backend  (reads .env: provider, DATABASE_URL, JWT_SECRET, ANTHROPIC_API_KEY)
python -m uvicorn industryiq.api.app:app --reload      # :8000

# 3) frontend
cd frontend ; npm install ; npm run dev                # :5173
```

**In the browser** (<http://localhost:5173>): register an email → start a chat →
watch the step timeline (`Thinking → Searching → Generating`) tick with a live
timer as the answer streams → expand sources → drag in a `.pdf`/`.docx`/`.txt`
and ask about it → rename and delete the chat → sign out and back in (your chats
persist). This is the only check that exercises the React UI, the live step
timers, and CORS together.

**Headless API smoke** (no browser) — auth → create → SSE stream → delete:

```powershell
$base = "http://localhost:8000"
$cred = @{ email = "e2e$(Get-Random)@example.com"; password = "password123" } | ConvertTo-Json
$tok  = (Invoke-RestMethod "$base/auth/register" -Method Post -ContentType application/json -Body $cred).access_token
$auth = @{ Authorization = "Bearer $tok" }
$cid  = (Invoke-RestMethod "$base/conversations" -Method Post -Headers $auth `
  -ContentType application/json -Body (@{ title = "e2e" } | ConvertTo-Json)).id
$body = @{ question = "Give a one-sentence overview." } | ConvertTo-Json
$txt  = (Invoke-WebRequest "$base/conversations/$cid/messages/stream" -Method Post -Headers $auth `
  -ContentType application/json -Body $body -TimeoutSec 150 -UseBasicParsing).Content
"frames -> status:$([bool]($txt -match 'event: status')) token:$([bool]($txt -match 'event: token')) done:$([bool]($txt -match 'event: done'))"
Invoke-WebRequest "$base/conversations/$cid" -Method Delete -Headers $auth -UseBasicParsing | % StatusCode  # 204
```

> Prerequisite for the live run: `pip install -e ".[dev,local]"` and `RAG_PROVIDER=anthropic`
> with `ANTHROPIC_API_KEY` set (real answers via Claude + a local embedder), or
> `RAG_PROVIDER=fake` for a no-key plumbing check.

## Stop and reset local services

Local Postgres (and the optional Milvus stack) run under Docker Compose. If you ran
the API in the foreground (host venv), **Ctrl+C** stops it; if you started it with
[`./scripts/start.ps1`](scripts/start.ps1) it runs as a container, stopped by
`docker compose down` below.

```powershell
docker compose down       # stop + remove containers (named data volumes survive)
docker compose down -v    # also delete volumes -> wipes the DB (users, chats, ingested docs)
docker compose ps         # see what's still running
```

- Use `down` between sessions to free resources while **keeping** your ingested
  knowledge base and accounts.
- Use `down -v` for a clean slate — e.g. when switching embedding providers, since a
  384-dim local embedder and 1024-dim Bedrock Titan can't share one pgvector table.

> This section is **local only**. Deploying to / stopping / tearing down the **AWS**
> environment is covered entirely in **[DEPLOY.md](DEPLOY.md)** (see its
> [Tear down](DEPLOY.md#7-tear-down) section).
