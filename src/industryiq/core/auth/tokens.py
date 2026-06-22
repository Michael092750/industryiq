"""Stateless access tokens -- signed JWTs (HS256).

The token carries only the user id (``sub``) and an expiry; nothing secret.
Decoding verifies signature and expiry, returning the subject or ``None`` for any
invalid/expired/tampered token -- the caller turns ``None`` into a 401. Keeping
both halves here means encode and decode can never disagree on the claim set.
"""

from datetime import UTC, datetime, timedelta

import jwt


def encode_token(
    user_id: str,
    *,
    secret: str,
    algorithm: str,
    expiry_minutes: int,
    now: datetime | None = None,
) -> str:
    """Mint a signed token for ``user_id`` that expires after ``expiry_minutes``."""
    issued = now or datetime.now(UTC)
    payload = {
        "sub": user_id,
        "iat": issued,
        "exp": issued + timedelta(minutes=expiry_minutes),
    }
    token: str = jwt.encode(payload, secret, algorithm=algorithm)
    return token


def decode_token(token: str, *, secret: str, algorithm: str) -> str | None:
    """Return the subject (user id) of a valid token, or ``None`` if it is not."""
    try:
        payload = jwt.decode(token, secret, algorithms=[algorithm])
    except jwt.PyJWTError:
        return None
    subject = payload.get("sub")
    return subject if isinstance(subject, str) else None
