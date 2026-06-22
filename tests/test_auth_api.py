from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from industryiq.api.app import app
from industryiq.api.deps import get_auth_service
from industryiq.core.auth import AuthService, InMemoryUserStore


@pytest.fixture
def client() -> Iterator[TestClient]:
    # A fresh in-memory auth service per test (empty user store, fixed secret).
    auth = AuthService(
        InMemoryUserStore(), secret="test-secret-key-of-sufficient-length", expiry_minutes=60
    )
    app.dependency_overrides[get_auth_service] = lambda: auth
    yield TestClient(app)
    app.dependency_overrides.clear()


def _register(client: TestClient, email: str = "a@example.com", password: str = "password123"):
    return client.post("/auth/register", json={"email": email, "password": password})


def test_register_returns_token(client: TestClient) -> None:
    response = _register(client)
    assert response.status_code == 201
    body = response.json()
    assert body["access_token"]
    assert body["token_type"] == "bearer"


def test_register_duplicate_email_is_409(client: TestClient) -> None:
    _register(client)
    assert _register(client).status_code == 409


def test_register_rejects_bad_email(client: TestClient) -> None:
    assert _register(client, email="not-an-email").status_code == 422


def test_register_rejects_short_password(client: TestClient) -> None:
    assert _register(client, password="short").status_code == 422


def test_login_with_correct_credentials(client: TestClient) -> None:
    _register(client)
    response = client.post(
        "/auth/login", json={"email": "a@example.com", "password": "password123"}
    )
    assert response.status_code == 200
    assert response.json()["access_token"]


def test_login_with_wrong_password_is_401(client: TestClient) -> None:
    _register(client)
    response = client.post("/auth/login", json={"email": "a@example.com", "password": "nope12345"})
    assert response.status_code == 401


def test_login_unknown_email_is_401(client: TestClient) -> None:
    response = client.post(
        "/auth/login", json={"email": "ghost@example.com", "password": "pw123456"}
    )
    assert response.status_code == 401


def test_me_requires_authentication(client: TestClient) -> None:
    assert client.get("/auth/me").status_code == 401


def test_me_returns_current_user(client: TestClient) -> None:
    token = _register(client).json()["access_token"]
    response = client.get("/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 200
    assert response.json()["email"] == "a@example.com"


def test_me_rejects_garbage_token(client: TestClient) -> None:
    response = client.get("/auth/me", headers={"Authorization": "Bearer not-a-real-token"})
    assert response.status_code == 401


def test_protected_conversation_route_requires_token(client: TestClient) -> None:
    # The chat surface is now gated; without a token it is 401, not 200.
    assert client.post("/conversations", json={"title": "x"}).status_code == 401
