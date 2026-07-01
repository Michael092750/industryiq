from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from industryiq.api.app import app
from industryiq.api.deps import get_pipeline
from industryiq.core.embeddings import FakeEmbedder
from industryiq.core.generation import FakeLLM
from industryiq.core.pipeline import RagPipeline
from industryiq.core.retrieval import Retriever
from industryiq.core.vectorstore import InMemoryVectorStore

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
    monkeypatch.setattr("industryiq.api.deps.PgVectorStore", FakePg)
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
    monkeypatch.setattr("industryiq.core.bedrock.BedrockEmbedder", FakeBedrockEmbedder)
    monkeypatch.setattr("industryiq.core.bedrock.BedrockLLM", FakeBedrockLLM)
    get_pipeline.cache_clear()
    assert isinstance(get_pipeline(), RagPipeline)
    get_pipeline.cache_clear()


def test_get_pipeline_uses_anthropic_when_provider_is_anthropic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Stub the local provider classes so no model download / API client happens.
    class FakeLocalEmbedder:
        @property
        def dim(self) -> int:
            return 384

    class FakeAnthropicLLM:
        def __init__(self, model_id: str, api_key: str | None) -> None: ...

        def generate(self, prompt: str) -> str:
            return "local-ish"

    monkeypatch.setenv("RAG_PROVIDER", "anthropic")
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setattr("industryiq.core.anthropic_llm.AnthropicLLM", FakeAnthropicLLM)
    monkeypatch.setattr("industryiq.core.local_embeddings.LocalEmbedder", FakeLocalEmbedder)
    get_pipeline.cache_clear()
    assert isinstance(get_pipeline(), RagPipeline)
    get_pipeline.cache_clear()


def test_admin_ingest_pdf(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ADMIN_API_KEY", "adm1n")
    pdf = FIXTURES / "sample.pdf"
    with pdf.open("rb") as handle:
        response = client.post(
            "/admin/ingest",
            files={"files": ("sample.pdf", handle, "application/pdf")},
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
                "files": (
                    "sample.docx",
                    handle,
                    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                )
            },
            headers={"X-Admin-Key": "adm1n"},
        )
    assert response.status_code == 200
    assert response.json()["chunk_ids"]


def test_admin_ingest_tags_optional_metadata(
    client: TestClient, pipeline: RagPipeline, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The optional form fields ride along on every chunk, mirroring the bulk path,
    # so an admin upload is faceted/attributable just like a bulk-ingested file.
    monkeypatch.setenv("ADMIN_API_KEY", "adm1n")
    response = client.post(
        "/admin/ingest",
        files={"files": ("a.txt", b"hello world from an admin upload", "text/plain")},
        data={
            "category": "AI",
            "publisher": "mckinsey.com",
            "source_type": "consultancy",
            "published_date": "2024",
        },
        headers={"X-Admin-Key": "adm1n"},
    )
    assert response.status_code == 200
    meta = pipeline.list_chunks()[0][1]
    assert meta["category"] == "AI"
    assert meta["publisher"] == "mckinsey.com"
    assert meta["source_type"] == "consultancy"
    assert meta["published_date"] == "2024"


def test_admin_ingest_omits_blank_metadata(
    client: TestClient, pipeline: RagPipeline, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A blank/whitespace field is treated as unset, not stored as an empty value
    # (which in Milvus would still create the promoted column).
    monkeypatch.setenv("ADMIN_API_KEY", "adm1n")
    response = client.post(
        "/admin/ingest",
        files={"files": ("a.txt", b"hello world from an admin upload", "text/plain")},
        data={"category": "AI", "publisher": "   "},
        headers={"X-Admin-Key": "adm1n"},
    )
    assert response.status_code == 200
    meta = pipeline.list_chunks()[0][1]
    assert meta["category"] == "AI"
    assert "publisher" not in meta
    assert "source_type" not in meta
    assert "published_date" not in meta


def test_admin_ingest_multiple_files(
    client: TestClient, pipeline: RagPipeline, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A batch ingests every file and applies the shared metadata to each; the
    # response breaks the chunks down per file and aggregates them in chunk_ids.
    monkeypatch.setenv("ADMIN_API_KEY", "adm1n")
    response = client.post(
        "/admin/ingest",
        files=[
            ("files", ("a.txt", b"first report about widgets", "text/plain")),
            ("files", ("b.txt", b"second report about gadgets", "text/plain")),
        ],
        data={"category": "AI"},
        headers={"X-Admin-Key": "adm1n"},
    )
    assert response.status_code == 200
    body = response.json()
    assert {f["filename"] for f in body["files"]} == {"a.txt", "b.txt"}
    assert len(body["chunk_ids"]) == sum(len(f["chunk_ids"]) for f in body["files"])
    # The shared tag lands on every chunk of every file.
    assert all(meta["category"] == "AI" for _, meta in pipeline.list_chunks())


def test_admin_ingest_rejects_whole_batch_on_bad_type(
    client: TestClient, pipeline: RagPipeline, monkeypatch: pytest.MonkeyPatch
) -> None:
    # One unsupported file rejects the batch (415) before anything is ingested,
    # so a batch never lands half-applied.
    monkeypatch.setenv("ADMIN_API_KEY", "adm1n")
    response = client.post(
        "/admin/ingest",
        files=[
            ("files", ("good.txt", b"a supported file", "text/plain")),
            ("files", ("data.csv", b"a,b,c", "text/csv")),
        ],
        headers={"X-Admin-Key": "adm1n"},
    )
    assert response.status_code == 415
    assert pipeline.list_chunks() == []  # nothing ingested


def test_admin_ingest_manifest_sets_per_file_metadata(
    client: TestClient, pipeline: RagPipeline, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A manifest gives each file its own metadata, matched to files by order.
    monkeypatch.setenv("ADMIN_API_KEY", "adm1n")
    manifest = (
        b"category,publisher,source_type,published_date\n"
        b"AI,mckinsey.com,consultancy,2024\n"
        b"Finance,bcg.com,consultancy,2023\n"
    )
    response = client.post(
        "/admin/ingest",
        files=[
            ("files", ("a.txt", b"first report about widgets", "text/plain")),
            ("files", ("b.txt", b"second report about gadgets", "text/plain")),
            ("manifest", ("manifest.csv", manifest, "text/csv")),
        ],
        headers={"X-Admin-Key": "adm1n"},
    )
    assert response.status_code == 200
    by_source = {meta["source"]: meta for _, meta in pipeline.list_chunks()}
    assert by_source["a.txt"]["category"] == "AI"
    assert by_source["a.txt"]["publisher"] == "mckinsey.com"
    assert by_source["a.txt"]["published_date"] == "2024"
    assert by_source["b.txt"]["category"] == "Finance"
    assert by_source["b.txt"]["publisher"] == "bcg.com"
    assert by_source["b.txt"]["published_date"] == "2023"


def test_admin_ingest_manifest_cell_falls_back_to_inline_default(
    client: TestClient, pipeline: RagPipeline, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A filled manifest cell wins; a blank cell (or a column absent from the
    # manifest) falls back to the like-named inline default.
    monkeypatch.setenv("ADMIN_API_KEY", "adm1n")
    manifest = b"category,publisher\nAI,mckinsey.com\n,bcg.com\n"  # row 2 leaves category blank
    response = client.post(
        "/admin/ingest",
        files=[
            ("files", ("a.txt", b"first report", "text/plain")),
            ("files", ("b.txt", b"second report", "text/plain")),
            ("manifest", ("m.csv", manifest, "text/csv")),
        ],
        data={"category": "Default", "source_type": "consultancy"},
        headers={"X-Admin-Key": "adm1n"},
    )
    assert response.status_code == 200
    by_source = {meta["source"]: meta for _, meta in pipeline.list_chunks()}
    assert by_source["a.txt"]["category"] == "AI"  # manifest cell wins
    assert by_source["a.txt"]["source_type"] == "consultancy"  # column absent -> default fills
    assert by_source["b.txt"]["category"] == "Default"  # blank cell -> default
    assert by_source["b.txt"]["publisher"] == "bcg.com"
    assert by_source["b.txt"]["source_type"] == "consultancy"


def test_admin_ingest_manifest_row_count_mismatch_is_400(
    client: TestClient, pipeline: RagPipeline, monkeypatch: pytest.MonkeyPatch
) -> None:
    # One row for two files is rejected (400) before anything is ingested.
    monkeypatch.setenv("ADMIN_API_KEY", "adm1n")
    response = client.post(
        "/admin/ingest",
        files=[
            ("files", ("a.txt", b"first", "text/plain")),
            ("files", ("b.txt", b"second", "text/plain")),
            ("manifest", ("m.csv", b"category\nAI\n", "text/csv")),
        ],
        headers={"X-Admin-Key": "adm1n"},
    )
    assert response.status_code == 400
    assert pipeline.list_chunks() == []  # nothing ingested


def test_admin_ingest_manifest_unrecognized_columns_is_400(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A header with none of the recognized columns is a wrong-format upload (400).
    monkeypatch.setenv("ADMIN_API_KEY", "adm1n")
    response = client.post(
        "/admin/ingest",
        files=[
            ("files", ("a.txt", b"first", "text/plain")),
            ("manifest", ("m.csv", b"foo,bar\n1,2\n", "text/csv")),
        ],
        headers={"X-Admin-Key": "adm1n"},
    )
    assert response.status_code == 400


def test_admin_ingest_unsupported_type_is_415(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ADMIN_API_KEY", "adm1n")
    response = client.post(
        "/admin/ingest",
        files={"files": ("data.csv", b"a,b,c", "text/csv")},
        headers={"X-Admin-Key": "adm1n"},
    )
    assert response.status_code == 415


def test_admin_ingest_rejects_missing_or_wrong_key(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ADMIN_API_KEY", "adm1n")
    files = {"files": ("a.txt", b"hi", "text/plain")}
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
        files={"files": ("a.txt", b"hi", "text/plain")},
        headers={"X-Admin-Key": "anything"},
    )
    assert response.status_code == 404


def test_admin_ui_page_is_served(client: TestClient) -> None:
    response = client.get("/admin/ui")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "shared knowledge base" in response.text
    assert "admin key" in response.text  # the key field is on the page
    assert "meta-category" in response.text  # optional-metadata fields are on the page
    assert 'id="manifest"' in response.text  # manifest upload input
    assert "category,publisher,source_type,published_date" in response.text  # manifest example
    assert 'id="file-list"' in response.text  # ordered, reorderable file list


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
