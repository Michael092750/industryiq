"""Domain types for authentication.

A single immutable value object shared across the auth module. ``password_hash``
is a bcrypt hash, never the plaintext; the domain carries it because the service
needs it to verify a login, and routes are responsible for never serializing it
back to a client.
"""

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class User:
    """A registered account: an id, the login email, and the password hash."""

    id: str
    email: str
    password_hash: str
    created_at: datetime
