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
> answers are placeholders until the real Bedrock/pgvector providers are wired
> in. Retrieval and source attribution are fully functional.
