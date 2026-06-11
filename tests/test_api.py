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


def test_default_pipeline_is_constructed() -> None:
    # Exercises the real wiring (not the test override).
    assert isinstance(get_pipeline(), RagPipeline)


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


def test_ingest_file_unsupported_type_is_415(client: TestClient) -> None:
    response = client.post(
        "/ingest/file",
        files={"file": ("data.csv", b"a,b,c", "text/csv")},
    )
    assert response.status_code == 415


def test_query_missing_question_is_422(client: TestClient) -> None:
    assert client.post("/query", json={}).status_code == 422


def test_query_empty_question_is_422(client: TestClient) -> None:
    assert client.post("/query", json={"question": ""}).status_code == 422


def test_query_on_empty_store_returns_answer_with_no_sources(client: TestClient) -> None:
    response = client.post("/query", json={"question": "anything"})
    assert response.status_code == 200
    assert response.json()["sources"] == []
