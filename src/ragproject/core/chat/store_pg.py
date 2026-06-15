"""A Postgres implementation of the ConversationStore interface.

Satisfies the same :class:`ragproject.core.chat.ports.ConversationStore` protocol
as :class:`InMemoryConversationStore`, so the service and API are unchanged. It
follows the conventions of :class:`ragproject.core.pgvectorstore.PgVectorStore`:
raw ``psycopg``, tables created on construction with ``CREATE TABLE IF NOT
EXISTS``, and one connection per operation for thread-safety under the web server.

A monotonic ``seq`` column gives turns a stable insertion order independent of
wall-clock timestamps.
"""

import uuid
from datetime import UTC, datetime
from typing import Any

import psycopg

from ragproject.core.chat.models import Conversation, Turn


class PgConversationStore:
    """Conversation store backed by two Postgres tables."""

    def __init__(
        self,
        dsn: str,
        *,
        conversations_table: str = "conversations",
        messages_table: str = "messages",
    ) -> None:
        self._dsn = dsn
        self._conversations_table = conversations_table
        self._messages_table = messages_table
        with psycopg.connect(dsn) as conn:
            conn.execute(
                f"CREATE TABLE IF NOT EXISTS {conversations_table} ("
                f"id TEXT PRIMARY KEY, title TEXT NOT NULL, "
                f"created_at TIMESTAMPTZ NOT NULL DEFAULT now())"
            )
            conn.execute(
                f"CREATE TABLE IF NOT EXISTS {messages_table} ("
                f"seq BIGSERIAL PRIMARY KEY, conversation_id TEXT NOT NULL, "
                f"question TEXT NOT NULL, answer TEXT NOT NULL, "
                f"created_at TIMESTAMPTZ NOT NULL DEFAULT now())"
            )
            conn.execute(
                f"CREATE INDEX IF NOT EXISTS {messages_table}_conv_seq_idx "
                f"ON {messages_table} (conversation_id, seq)"
            )
            conn.commit()

    def create(self, title: str) -> Conversation:
        conversation = Conversation(
            id=uuid.uuid4().hex[:16],
            title=title,
            created_at=datetime.now(UTC),
        )
        with psycopg.connect(self._dsn) as conn:
            conn.execute(
                f"INSERT INTO {self._conversations_table} (id, title, created_at) "
                f"VALUES (%s, %s, %s)",
                (conversation.id, conversation.title, conversation.created_at),
            )
            conn.commit()
        return conversation

    def get(self, conversation_id: str) -> Conversation | None:
        with psycopg.connect(self._dsn) as conn:
            row = conn.execute(
                f"SELECT id, title, created_at FROM {self._conversations_table} WHERE id = %s",
                (conversation_id,),
            ).fetchone()
        if row is None:
            return None
        return Conversation(id=row[0], title=row[1], created_at=row[2])

    def history(self, conversation_id: str, limit: int | None = None) -> list[Turn]:
        params: tuple[Any, ...] = (conversation_id,)
        if limit is None:
            sql = (
                f"SELECT question, answer FROM {self._messages_table} "
                f"WHERE conversation_id = %s ORDER BY seq"
            )
        else:
            # Take the most recent `limit` turns, then return them oldest-first.
            sql = (
                f"SELECT question, answer FROM ("
                f"SELECT question, answer, seq FROM {self._messages_table} "
                f"WHERE conversation_id = %s ORDER BY seq DESC LIMIT %s"
                f") sub ORDER BY seq"
            )
            params = (conversation_id, limit)
        with psycopg.connect(self._dsn) as conn:
            rows = conn.execute(sql, params).fetchall()
        return [Turn(question=row[0], answer=row[1]) for row in rows]

    def append(self, conversation_id: str, turn: Turn) -> None:
        with psycopg.connect(self._dsn) as conn:
            conn.execute(
                f"INSERT INTO {self._messages_table} (conversation_id, question, answer) "
                f"VALUES (%s, %s, %s)",
                (conversation_id, turn.question, turn.answer),
            )
            conn.commit()
