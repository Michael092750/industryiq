"""In-memory user store: the default (no DATABASE_URL) and the test double.

Dict-backed, mirroring :class:`InMemoryConversationStore`. It satisfies the
:class:`industryiq.core.auth.ports.UserStore` port, so the auth service cannot
tell it apart from the Postgres-backed store. Emails arrive already normalized
from the service, so the by-email index keys on them verbatim.
"""

import uuid
from datetime import UTC, datetime

from industryiq.core.auth.models import User
from industryiq.core.auth.ports import UserStore


class InMemoryUserStore(UserStore):
    """A dict-backed user store for tests and local development."""

    def __init__(self) -> None:
        self._by_id: dict[str, User] = {}
        self._by_email: dict[str, User] = {}

    def create(self, email: str, password_hash: str) -> User:
        user = User(
            id=uuid.uuid4().hex[:16],
            email=email,
            password_hash=password_hash,
            created_at=datetime.now(UTC),
        )
        self._by_id[user.id] = user
        self._by_email[user.email] = user
        return user

    def get_by_email(self, email: str) -> User | None:
        return self._by_email.get(email)

    def get_by_id(self, user_id: str) -> User | None:
        return self._by_id.get(user_id)
