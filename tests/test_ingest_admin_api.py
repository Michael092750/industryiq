"""API tests for the scheduled-ingestion admin endpoints.

Drive the real routes with a TestClient, overriding ``get_ingestion_service`` with
a service backed by offline fakes + in-memory stores. The client is built without
the lifespan context manager (like the other API tests), so the real background
scheduler never starts.
"""

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from industryiq.api.app import app
from industryiq.api.deps import get_ingestion_service
from industryiq.core.embeddings import FakeEmbedder
from industryiq.core.generation import FakeLLM
from industryiq.core.ingestion import IngestionService, InMemoryIngestStateStore
from industryiq.core.pipeline import RagPipeline
from industryiq.core.retrieval import Retriever
from industryiq.core.vectorstore import InMemoryVectorStore


@pytest.fixture
def service() -> IngestionService:
    pipeline = RagPipeline(Retriever(FakeEmbedder(dim=16), InMemoryVectorStore()), FakeLLM())
    return IngestionService(pipeline, InMemoryIngestStateStore())


@pytest.fixture
def client(service: IngestionService) -> Iterator[TestClient]:
    app.dependency_overrides[get_ingestion_service] = lambda: service
    yield TestClient(app)
    app.dependency_overrides.clear()


ADMIN = {"X-Admin-Key": "adm1n"}


def test_get_ingest_job_returns_default_config(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ADMIN_API_KEY", "adm1n")
    response = client.get("/admin/ingest-job", headers=ADMIN)
    assert response.status_code == 200
    body = response.json()
    assert body["config"]["enabled"] is False
    assert body["config"]["path"] is None
    assert body["last_run"] is None


def test_put_then_get_round_trips_config(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ADMIN_API_KEY", "adm1n")
    put = client.put(
        "/admin/ingest-job",
        headers=ADMIN,
        json={"path": "/data/reports", "interval_minutes": 30, "enabled": True},
    )
    assert put.status_code == 200
    assert put.json()["path"] == "/data/reports"

    got = client.get("/admin/ingest-job", headers=ADMIN).json()
    assert got["config"] == {
        "path": "/data/reports",
        "interval_minutes": 30,
        "enabled": True,
        "updated_at": got["config"]["updated_at"],  # set by the server
    }
    assert got["config"]["updated_at"] is not None


def test_put_rejects_interval_below_one(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ADMIN_API_KEY", "adm1n")
    response = client.put(
        "/admin/ingest-job",
        headers=ADMIN,
        json={"path": "/data", "interval_minutes": 0, "enabled": True},
    )
    assert response.status_code == 422


def test_run_now_ingests_configured_path(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("ADMIN_API_KEY", "adm1n")
    (tmp_path / "AI").mkdir()
    (tmp_path / "AI" / "a.txt").write_text("hello world", encoding="utf-8")
    client.put(
        "/admin/ingest-job",
        headers=ADMIN,
        json={"path": str(tmp_path), "interval_minutes": 60, "enabled": True},
    )

    response = client.post("/admin/ingest-job/run-now", headers=ADMIN)
    assert response.status_code == 200
    body = response.json()
    assert body["ingested"] == 1
    assert body["by_category"] == {"AI": body["chunks_added"]}
    assert body["busy"] is False


def test_run_now_without_path_is_400(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ADMIN_API_KEY", "adm1n")
    response = client.post("/admin/ingest-job/run-now", headers=ADMIN)
    assert response.status_code == 400


@pytest.mark.parametrize(
    "method, path",
    [
        ("get", "/admin/ingest-job"),
        ("put", "/admin/ingest-job"),
        ("post", "/admin/ingest-job/run-now"),
    ],
)
def test_endpoints_reject_missing_or_wrong_key(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, method: str, path: str
) -> None:
    monkeypatch.setenv("ADMIN_API_KEY", "adm1n")
    # request() (not get/post) so a body can ride along uniformly across methods.
    assert client.request(method, path, json={}).status_code == 401
    wrong = client.request(method, path, json={}, headers={"X-Admin-Key": "wrong"})
    assert wrong.status_code == 401


@pytest.mark.parametrize(
    "method, path",
    [
        ("get", "/admin/ingest-job"),
        ("put", "/admin/ingest-job"),
        ("post", "/admin/ingest-job/run-now"),
    ],
)
def test_endpoints_disabled_when_no_key_configured(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, method: str, path: str
) -> None:
    monkeypatch.delenv("ADMIN_API_KEY", raising=False)
    response = client.request(method, path, json={}, headers={"X-Admin-Key": "anything"})
    assert response.status_code == 404


def test_admin_ui_has_schedule_panel(client: TestClient) -> None:
    text = client.get("/admin/ui").text
    assert "Scheduled bulk ingestion" in text
    assert "Run now" in text
