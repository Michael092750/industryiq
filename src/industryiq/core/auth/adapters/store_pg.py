"""A Postgres implementation of the UserStore interface.

Follows the conventions of :class:`industryiq.core.chat.adapters.store_pg`: raw
``psycopg``, the table created on construction with ``CREATE TABLE IF NOT
EXISTS``, and one connection per operation for thread-safety under the web
server. ``email`` is stored ``UNIQUE`` (the database is the final guard against a
duplicate even under a registration race) and already normalized by the service.
"""

import uuid
from datetime import UTC, datetime
from typing import Any

import psycopg

from industryiq.core.auth.models import User
from industryiq.core.auth.ports import UserStore


def _row_to_user(row: tuple[Any, ...] | None) -> User | None:
    if row is None:
        return None
    return User(id=row[0], email=row[1], password_hash=row[2], created_at=row[3])


class PgUserStore(UserStore):
    """User store backed by a single Postgres table."""

    _COLUMNS = "id, email, password_hash, created_at"

    def __init__(self, dsn: str, *, users_table: str = "users") -> None:
        self._dsn = dsn
        self._users_table = users_table
        with psycopg.connect(dsn) as conn:
            conn.execute(
                f"CREATE TABLE IF NOT EXISTS {users_table} ("
                f"id TEXT PRIMARY KEY, email TEXT UNIQUE NOT NULL, "
                f"password_hash TEXT NOT NULL, "
                f"created_at TIMESTAMPTZ NOT NULL DEFAULT now())"
            )
            conn.commit()

    def create(self, email: str, password_hash: str) -> User:
        user = User(
            id=uuid.uuid4().hex[:16],
            email=email,
            password_hash=password_hash,
            created_at=datetime.now(UTC),
        )
        with psycopg.connect(self._dsn) as conn:
            conn.execute(
                f"INSERT INTO {self._users_table} ({self._COLUMNS}) VALUES (%s, %s, %s, %s)",
                (user.id, user.email, user.password_hash, user.created_at),
            )
            conn.commit()
        return user

    def get_by_email(self, email: str) -> User | None:
        with psycopg.connect(self._dsn) as conn:
            row = conn.execute(
                f"SELECT {self._COLUMNS} FROM {self._users_table} WHERE email = %s",
                (email,),
            ).fetchone()
        return _row_to_user(row)

    def get_by_id(self, user_id: str) -> User | None:
        with psycopg.connect(self._dsn) as conn:
            row = conn.execute(
                f"SELECT {self._COLUMNS} FROM {self._users_table} WHERE id = %s",
                (user_id,),
            ).fetchone()
        return _row_to_user(row)
