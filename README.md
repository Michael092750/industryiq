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

Endpoints:

- `GET /health` — liveness check
- `POST /ingest` — body `{"text": "...", "source": "optional"}`
- `POST /query` — body `{"question": "...", "k": 5}` → `{"answer", "sources"}`

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
