"""A Postgres implementation of the ConversationStore interface.

Satisfies the same :class:`industryiq.core.chat.ports.ConversationStore` protocol
as :class:`InMemoryConversationStore`, so the service and API are unchanged. It
follows the conventions of :class:`industryiq.core.pgvectorstore.PgVectorStore`:
raw ``psycopg``, tables created on construction with ``CREATE TABLE IF NOT
EXISTS``, and one connection per operation for thread-safety under the web server.

A monotonic ``seq`` column gives turns a stable insertion order independent of
wall-clock timestamps.
"""

import uuid
from datetime import UTC, datetime
from typing import Any

import psycopg

from industryiq.core.chat.models import Conversation, Turn
from industryiq.core.chat.ports import ConversationStore


class PgConversationStore(ConversationStore):
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
                f"id TEXT PRIMARY KEY, title TEXT NOT NULL, owner_id TEXT, "
                f"created_at TIMESTAMPTZ NOT NULL DEFAULT now())"
            )
            # Migrate a pre-auth table in place: add owner_id if it predates it.
            # Existing rows keep owner_id NULL and become invisible to all users.
            conn.execute(
                f"ALTER TABLE {conversations_table} ADD COLUMN IF NOT EXISTS owner_id TEXT"
            )
            conn.execute(
                f"CREATE INDEX IF NOT EXISTS {conversations_table}_owner_idx "
                f"ON {conversations_table} (owner_id)"
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

    def create(self, title: str, owner_id: str | None = None) -> Conversation:
        conversation = Conversation(
            id=uuid.uuid4().hex[:16],
            title=title,
            created_at=datetime.now(UTC),
            owner_id=owner_id,
        )
        with psycopg.connect(self._dsn) as conn:
            conn.execute(
                f"INSERT INTO {self._conversations_table} (id, title, owner_id, created_at) "
                f"VALUES (%s, %s, %s, %s)",
                (
                    conversation.id,
                    conversation.title,
                    conversation.owner_id,
                    conversation.created_at,
                ),
            )
            conn.commit()
        return conversation

    def get(self, conversation_id: str) -> Conversation | None:
        with psycopg.connect(self._dsn) as conn:
            row = conn.execute(
                f"SELECT id, title, created_at, owner_id FROM {self._conversations_table} "
                f"WHERE id = %s",
                (conversation_id,),
            ).fetchone()
        if row is None:
            return None
        return Conversation(id=row[0], title=row[1], created_at=row[2], owner_id=row[3])

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

    def rename(self, conversation_id: str, title: str) -> None:
        with psycopg.connect(self._dsn) as conn:
            conn.execute(
                f"UPDATE {self._conversations_table} SET title = %s WHERE id = %s",
                (title, conversation_id),
            )
            conn.commit()

    def delete(self, conversation_id: str) -> None:
        # No FK/cascade between the tables, so remove the messages explicitly.
        with psycopg.connect(self._dsn) as conn:
            conn.execute(
                f"DELETE FROM {self._messages_table} WHERE conversation_id = %s",
                (conversation_id,),
            )
            conn.execute(
                f"DELETE FROM {self._conversations_table} WHERE id = %s",
                (conversation_id,),
            )
            conn.commit()

    def list_all(self, owner_id: str | None = None) -> list[Conversation]:
        base = f"SELECT id, title, created_at, owner_id FROM {self._conversations_table}"
        params: tuple[Any, ...] = ()
        if owner_id is not None:
            base += " WHERE owner_id = %s"
            params = (owner_id,)
        base += " ORDER BY created_at DESC"
        with psycopg.connect(self._dsn) as conn:
            rows = conn.execute(base, params).fetchall()
        return [
            Conversation(id=row[0], title=row[1], created_at=row[2], owner_id=row[3])
            for row in rows
        ]
