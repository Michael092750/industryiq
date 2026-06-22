import pytest

from industryiq.core.auth import AuthService, EmailAlreadyRegistered, InMemoryUserStore


def _service() -> AuthService:
    return AuthService(
        InMemoryUserStore(), secret="test-secret-key-of-sufficient-length", expiry_minutes=60
    )


def test_register_creates_account() -> None:
    auth = _service()
    user = auth.register("a@example.com", "password123")
    assert user.email == "a@example.com"
    assert user.password_hash != "password123"  # stored hashed, never plaintext


def test_register_normalizes_email() -> None:
    auth = _service()
    auth.register("  MixedCase@Example.COM ", "password123")
    # Login with any casing/whitespace resolves to the same account.
    assert auth.authenticate("mixedcase@example.com", "password123") is not None


def test_register_duplicate_email_raises() -> None:
    auth = _service()
    auth.register("a@example.com", "password123")
    with pytest.raises(EmailAlreadyRegistered):
        auth.register("A@example.com", "another-password")


def test_authenticate_accepts_correct_password() -> None:
    auth = _service()
    registered = auth.register("a@example.com", "password123")
    assert auth.authenticate("a@example.com", "password123") == registered


def test_authenticate_rejects_wrong_password() -> None:
    auth = _service()
    auth.register("a@example.com", "password123")
    assert auth.authenticate("a@example.com", "wrong-password") is None


def test_authenticate_unknown_email_returns_none() -> None:
    assert _service().authenticate("nobody@example.com", "password123") is None


def test_token_round_trips_to_user() -> None:
    auth = _service()
    user = auth.register("a@example.com", "password123")
    token = auth.create_token(user)
    assert auth.identify(token) == user


def test_identify_rejects_invalid_token() -> None:
    assert _service().identify("not-a-valid-token") is None
