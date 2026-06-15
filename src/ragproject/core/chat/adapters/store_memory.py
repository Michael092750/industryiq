"""In-memory conversation store: the default (no DATABASE_URL) and test double.

Dict-backed, mirroring :class:`ragproject.core.vectorstore.InMemoryVectorStore`.
It satisfies the :class:`ragproject.core.chat.ports.ConversationStore` port, so
the service cannot tell it apart from the Postgres-backed store.
"""

import uuid
from datetime import UTC, datetime

from ragproject.core.chat.models import Conversation, Turn


class InMemoryConversationStore:
    """A dict-backed conversation store for tests and local development."""

    def __init__(self) -> None:
        self._conversations: dict[str, Conversation] = {}
        self._turns: dict[str, list[Turn]] = {}

    def create(self, title: str) -> Conversation:
        conversation = Conversation(
            id=uuid.uuid4().hex[:16],
            title=title,
            created_at=datetime.now(UTC),
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
