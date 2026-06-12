from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from ragproject.api.app import app
from ragproject.api.deps import get_pipeline
from ragproject.core.embeddings import FakeEmbedder
from ragproject.core.generation import FakeLLM
from ragproject.core.loaders import load_pdf
from ragproject.core.pipeline import RagPipeline
from ragproject.core.retrieval import Retriever
from ragproject.core.vectorstore import InMemoryVectorStore

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def client() -> Iterator[TestClient]:
    # Inject a fresh, isolated pipeline for each test via dependency override.
    pipeline = RagPipeline(
        Retriever(FakeEmbedder(dim=16), InMemoryVectorStore()),
        FakeLLM(response="Grounded answer [1]."),
    )
    app.dependency_overrides[get_pipeline] = lambda: pipeline
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_health() -> None:
    assert TestClient(app).get("/health").json() == {"status": "ok"}


def test_get_pipeline_uses_in_memory_without_database_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RAG_PROVIDER", "fake")
    monkeypatch.delenv("DATABASE_URL", raising=False)
    get_pipeline.cache_clear()
    assert isinstance(get_pipeline(), RagPipeline)
    get_pipeline.cache_clear()


def test_get_pipeline_uses_pgvector_when_database_url_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorded: dict[str, object] = {}

    class FakePg:
        def __init__(self, dsn: str, dim: int) -> None:
            recorded["dsn"] = dsn
            recorded["dim"] = dim

    monkeypatch.setenv("RAG_PROVIDER", "fake")
    monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@host/db")
    monkeypatch.setattr("ragproject.api.deps.PgVectorStore", FakePg)
    get_pipeline.cache_clear()
    assert isinstance(get_pipeline(), RagPipeline)
    assert recorded["dsn"] == "postgresql://u:p@host/db"
    get_pipeline.cache_clear()


def test_get_pipeline_uses_bedrock_when_provider_is_bedrock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Stub the Bedrock classes so no AWS clients are constructed.
    class FakeBedrockEmbedder:
        def __init__(self, model_id: str, region: str) -> None: ...

        @property
        def dim(self) -> int:
            return 1024

    class FakeBedrockLLM:
        def __init__(self, model_id: str, region: str) -> None: ...

        def generate(self, prompt: str) -> str:
            return "real-ish"

    monkeypatch.setenv("RAG_PROVIDER", "bedrock")
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setattr("ragproject.core.bedrock.BedrockEmbedder", FakeBedrockEmbedder)
    monkeypatch.setattr("ragproject.core.bedrock.BedrockLLM", FakeBedrockLLM)
    get_pipeline.cache_clear()
    assert isinstance(get_pipeline(), RagPipeline)
    get_pipeline.cache_clear()


def test_ingest_then_query(client: TestClient) -> None:
    ingest = client.post("/ingest", json={"text": "cats are great and dogs are loyal"})
    assert ingest.status_code == 200
    assert len(ingest.json()["chunk_ids"]) == 1

    response = client.post("/query", json={"question": "cats are great and dogs are loyal"})
    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "Grounded answer [1]."
    assert body["sources"][0]["text"] == "cats are great and dogs are loyal"


def test_ingest_pdf_file_then_query(client: TestClient) -> None:
    pdf = FIXTURES / "sample.pdf"
    with pdf.open("rb") as handle:
        response = client.post(
            "/ingest/file",
            files={"file": ("sample.pdf", handle, "application/pdf")},
        )
    assert response.status_code == 200
    assert response.json()["chunk_ids"]

    # The stored chunk is the whitespace-normalized PDF text; query it verbatim.
    expected = " ".join(load_pdf(pdf).split())
    result = client.post("/query", json={"question": expected})
    assert result.json()["sources"][0]["text"] == expected


def test_ingest_docx_file(client: TestClient) -> None:
    docx = FIXTURES / "sample.docx"
    with docx.open("rb") as handle:
        response = client.post(
            "/ingest/file",
            files={
                "file": (
                    "sample.docx",
                    handle,
                    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                )
            },
        )
    assert response.status_code == 200
    assert response.json()["chunk_ids"]


def test_ingest_file_unsupported_type_is_415(client: TestClient) -> None:
    response = client.post(
        "/ingest/file",
        files={"file": ("data.csv", b"a,b,c", "text/csv")},
    )
    assert response.status_code == 415


def test_debug_chunks_shows_ingested_documents(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DEBUG_API_KEY", "s3cret")
    client.post("/ingest", json={"text": "first doc", "source": "a.txt"})
    client.post("/ingest", json={"text": "second doc", "source": "b.txt"})
    response = client.get("/debug/chunks", headers={"X-Debug-Key": "s3cret"})
    assert response.status_code == 200
    body = response.json()
    assert body["count"] == 2
    assert {chunk["source"] for chunk in body["chunks"]} == {"a.txt", "b.txt"}


def test_debug_chunks_rejects_missing_or_wrong_key(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DEBUG_API_KEY", "s3cret")
    assert client.get("/debug/chunks").status_code == 401
    assert client.get("/debug/chunks", headers={"X-Debug-Key": "wrong"}).status_code == 401


def test_debug_chunks_disabled_when_no_key_configured(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("DEBUG_API_KEY", raising=False)
    # Even with a header, the endpoint is invisible (404) when unconfigured.
    assert client.get("/debug/chunks", headers={"X-Debug-Key": "anything"}).status_code == 404


def test_debug_ui_page_is_served(client: TestClient) -> None:
    response = client.get("/debug-ui")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "Indexed chunks" in response.text


def test_query_missing_question_is_422(client: TestClient) -> None:
    assert client.post("/query", json={}).status_code == 422


def test_query_empty_question_is_422(client: TestClient) -> None:
    assert client.post("/query", json={"question": ""}).status_code == 422


def test_query_on_empty_store_returns_answer_with_no_sources(client: TestClient) -> None:
    response = client.post("/query", json={"question": "anything"})
    assert response.status_code == 200
    assert response.json()["sources"] == []
