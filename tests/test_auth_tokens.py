from datetime import UTC, datetime, timedelta

from industryiq.core.auth.tokens import decode_token, encode_token

# >= 32 bytes so pyjwt does not warn about a weak HMAC key.
SECRET = "unit-test-secret-key-of-sufficient-length"
ALGO = "HS256"


def test_round_trips_subject() -> None:
    token = encode_token("user-123", secret=SECRET, algorithm=ALGO, expiry_minutes=60)
    assert decode_token(token, secret=SECRET, algorithm=ALGO) == "user-123"


def test_expired_token_is_rejected() -> None:
    # Issued two hours ago with a one-hour lifetime -> already expired.
    issued = datetime.now(UTC) - timedelta(hours=2)
    token = encode_token("user-123", secret=SECRET, algorithm=ALGO, expiry_minutes=60, now=issued)
    assert decode_token(token, secret=SECRET, algorithm=ALGO) is None


def test_wrong_secret_is_rejected() -> None:
    token = encode_token("user-123", secret=SECRET, algorithm=ALGO, expiry_minutes=60)
    other_secret = "a-different-secret-of-sufficient-length"
    assert decode_token(token, secret=other_secret, algorithm=ALGO) is None


def test_tampered_token_is_rejected() -> None:
    token = encode_token("user-123", secret=SECRET, algorithm=ALGO, expiry_minutes=60)
    tampered = token[:-2] + ("aa" if token[-2:] != "aa" else "bb")
    assert decode_token(tampered, secret=SECRET, algorithm=ALGO) is None


def test_garbage_is_rejected() -> None:
    assert decode_token("not-a-jwt", secret=SECRET, algorithm=ALGO) is None
