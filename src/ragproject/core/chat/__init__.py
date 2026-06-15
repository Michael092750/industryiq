"""Multi-round chat: a RAG chatbot built as ports-and-adapters.

Public surface:

* :class:`ChatService` -- orchestrates a single conversational turn.
* Ports (:mod:`ragproject.core.chat.ports`) -- the abstractions it depends on.
* Adapters -- :class:`InMemoryConversationStore`, :class:`LlmQueryRewriter`,
  :class:`NoOpQueryRewriter`. The Postgres store lives in
  :mod:`ragproject.core.chat.store_pg` and is imported only where it is wired,
  to keep this package import light.
"""

from ragproject.core.chat.models import (
    ChatResult,
    Conversation,
    RouteDecision,
    StreamEnd,
    StreamEvent,
    StreamStart,
    StreamStatus,
    StreamToken,
    Turn,
)
from ragproject.core.chat.ports import (
    ConversationStore,
    QueryRewriter,
    RetrievalPort,
    RetrievalRouter,
)
from ragproject.core.chat.rewriting import LlmQueryRewriter, NoOpQueryRewriter
from ragproject.core.chat.routing import AlwaysRetrieveRouter, LlmRouter
from ragproject.core.chat.service import ChatService, ConversationNotFound
from ragproject.core.chat.store_memory import InMemoryConversationStore

__all__ = [
    "AlwaysRetrieveRouter",
    "ChatResult",
    "ChatService",
    "Conversation",
    "ConversationNotFound",
    "ConversationStore",
    "InMemoryConversationStore",
    "LlmQueryRewriter",
    "LlmRouter",
    "NoOpQueryRewriter",
    "QueryRewriter",
    "RetrievalPort",
    "RetrievalRouter",
    "RouteDecision",
    "StreamEnd",
    "StreamEvent",
    "StreamStart",
    "StreamStatus",
    "StreamToken",
    "Turn",
]
