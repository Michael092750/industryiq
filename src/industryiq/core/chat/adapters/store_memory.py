"""In-memory conversation store: the default (no DATABASE_URL) and test double.

Dict-backed, mirroring :class:`industryiq.core.vectorstore.InMemoryVectorStore`.
It satisfies the :class:`industryiq.core.chat.ports.ConversationStore` port, so
the service cannot tell it apart from the Postgres-backed store.
"""

import uuid
from dataclasses import replace
from datetime import UTC, datetime

from industryiq.core.chat.models import Conversation, Turn
from industryiq.core.chat.ports import ConversationStore


class InMemoryConversationStore(ConversationStore):
    """A dict-backed conversation store for tests and local development."""

    def __init__(self) -> None:
        self._conversations: dict[str, Conversation] = {}
        self._turns: dict[str, list[Turn]] = {}

    def create(self, title: str, owner_id: str | None = None) -> Conversation:
        conversation = Conversation(
            id=uuid.uuid4().hex[:16],
            title=title,
            created_at=datetime.now(UTC),
            owner_id=owner_id,
        )
        self._conversations[conversation.id] = conversation
        self._turns[conversation.id] = []
        return conversation

    def get(self, conversation_id: str) -> Conversation | None:
        return self._conversations.get(conversation_id)

    def history(self, conversation_id: str, limit: int | None = None) -> list[Turn]:
        turns = self._turns.get(conversation_id, [])
        return list(turns[-limit:]) if limit is not None else list(turns)

    def append(self, conversation_id: str, turn: Turn) -> None:
        self._turns.setdefault(conversation_id, []).append(turn)

    def rename(self, conversation_id: str, title: str) -> None:
        existing = self._conversations.get(conversation_id)
        if existing is not None:
            self._conversations[conversation_id] = replace(existing, title=title)

    def delete(self, conversation_id: str) -> None:
        self._conversations.pop(conversation_id, None)
        self._turns.pop(conversation_id, None)

    def list_all(self, owner_id: str | None = None) -> list[Conversation]:
        conversations = list(self._conversations.values())
        if owner_id is not None:
            conversations = [c for c in conversations if c.owner_id == owner_id]
        return sorted(conversations, key=lambda c: c.created_at, reverse=True)
