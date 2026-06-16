from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from ragproject.api.app import app
from ragproject.api.deps import get_pipeline
from ragproject.core.embeddings import FakeEmbedder
from ragproject.core.generation import FakeLLM
from ragproject.core.pipeline import RagPipeline
from ragproject.core.retrieval import Retriever
from ragproject.core.vectorstore import InMemoryVectorStore

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def pipeline() -> RagPipeline:
    return RagPipeline(
        Retriever(FakeEmbedder(dim=16), InMemoryVectorStore()),
        FakeLLM(response="Grounded answer [1]."),
    )


@pytest.fixture
def client(pipeline: RagPipeline) -> Iterator[TestClient]:
    # Inject the test pipeline so routes and the test share one store.
    app.dependency_overrides[get_pipeline] = lambda: pipeline
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_health() -> None:
    assert TestClient(app).get("/health").json() == {"status": "ok"}


def test_cors_allows_frontend_origin() -> None:
    response = TestClient(app).options(
        "/conversations",
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "POST",
        },
    )
    assert response.headers.get("access-control-allow-origin") == "http://localhost:5173"


def test_cors_allows_any_localhost_port() -> None:
    # Vite may pick a different port (5174, ...); the regex must allow it.
    response = TestClient(app).options(
        "/conversations",
        headers={
            "Origin": "http://localhost:5174",
            "Access-Control-Request-Method": "POST",
        },
    )
    assert response.headers.get("access-control-allow-origin") == "http://localhost:5174"


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


def test_admin_ingest_pdf(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ADMIN_API_KEY", "adm1n")
    pdf = FIXTURES / "sample.pdf"
    with pdf.open("rb") as handle:
        response = client.post(
            "/admin/ingest",
            files={"file": ("sample.pdf", handle, "application/pdf")},
            headers={"X-Admin-Key": "adm1n"},
        )
    assert response.status_code == 200
    assert response.json()["chunk_ids"]


def test_admin_ingest_docx(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ADMIN_API_KEY", "adm1n")
    docx = FIXTURES / "sample.docx"
    with docx.open("rb") as handle:
        response = client.post(
            "/admin/ingest",
            files={
                "file": (
                    "sample.docx",
                    handle,
                    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                )
            },
            headers={"X-Admin-Key": "adm1n"},
        )
    assert response.status_code == 200
    assert response.json()["chunk_ids"]


def test_admin_ingest_unsupported_type_is_415(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ADMIN_API_KEY", "adm1n")
    response = client.post(
        "/admin/ingest",
        files={"file": ("data.csv", b"a,b,c", "text/csv")},
        headers={"X-Admin-Key": "adm1n"},
    )
    assert response.status_code == 415


def test_admin_ingest_rejects_missing_or_wrong_key(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ADMIN_API_KEY", "adm1n")
    files = {"file": ("a.txt", b"hi", "text/plain")}
    assert client.post("/admin/ingest", files=files).status_code == 401
    assert (
        client.post("/admin/ingest", files=files, headers={"X-Admin-Key": "wrong"}).status_code
        == 401
    )


def test_admin_ingest_disabled_when_no_key_configured(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("ADMIN_API_KEY", raising=False)
    # Even with a header, the endpoint is invisible (404) when unconfigured.
    response = client.post(
        "/admin/ingest",
        files={"file": ("a.txt", b"hi", "text/plain")},
        headers={"X-Admin-Key": "anything"},
    )
    assert response.status_code == 404


def test_admin_ui_page_is_served(client: TestClient) -> None:
    response = client.get("/admin/ui")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "shared knowledge base" in response.text
    assert "admin key" in response.text  # the key field is on the page


def test_debug_chunks_shows_ingested_documents(
    client: TestClient, pipeline: RagPipeline, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DEBUG_API_KEY", "s3cret")
    pipeline.ingest_text("first doc", source="a.txt")
    pipeline.ingest_text("second doc", source="b.txt")
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
    assert "retrieval debug" in response.text
    assert "Retrieve" in response.text  # the query box / button


def test_debug_retrieve_ranks_chunks_for_query(
    client: TestClient, pipeline: RagPipeline, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DEBUG_API_KEY", "s3cret")
    pipeline.ingest_text("the sky is blue", source="facts.txt")
    response = client.get(
        "/debug/retrieve", params={"q": "the sky is blue"}, headers={"X-Debug-Key": "s3cret"}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["query"] == "the sky is blue"
    assert body["count"] >= 1
    top = body["chunks"][0]
    assert top["text"] == "the sky is blue"
    assert top["source"] == "facts.txt"
    assert isinstance(top["score"], int | float)


def test_debug_retrieve_requires_key(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEBUG_API_KEY", "s3cret")
    assert client.get("/debug/retrieve", params={"q": "x"}).status_code == 401


def test_debug_retrieve_empty_query_is_422(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DEBUG_API_KEY", "s3cret")
    response = client.get("/debug/retrieve", params={"q": ""}, headers={"X-Debug-Key": "s3cret"})
    assert response.status_code == 422
