"""The auth module's ports -- the abstractions it depends on.

Mirrors the chat module: :class:`AuthService` is written against this
``Protocol``, never a concrete store, so an in-memory fake (tests, local dev)
and Postgres (production) are interchangeable. Email is the unique natural key;
the service normalizes it (lowercased, trimmed) before any store call, so
implementations can compare it verbatim.
"""

from typing import Protocol, runtime_checkable

from industryiq.core.auth.models import User


@runtime_checkable
class UserStore(Protocol):
    """Persist user accounts, looked up by id or by (unique) email."""

    def create(self, email: str, password_hash: str) -> User: ...

    def get_by_email(self, email: str) -> User | None: ...

    def get_by_id(self, user_id: str) -> User | None: ...
