"""Multi-round chat: a RAG chatbot built as ports-and-adapters.

Public surface:

* :class:`ChatService` -- orchestrates a single conversational turn.
* Ports (:mod:`ragproject.core.chat.ports`) -- the abstractions it depends on.
* Adapters (:mod:`ragproject.core.chat.adapters`) -- the concrete
  implementations of the ports. The Postgres store is imported only where it is
  wired (:mod:`ragproject.api.deps`), to keep this package import light.
"""

from ragproject.core.chat.adapters.filtering import ThresholdFilter
from ragproject.core.chat.adapters.rewriting import LlmQueryRewriter, NoOpQueryRewriter
from ragproject.core.chat.adapters.routing import AlwaysRetrieveRouter, LlmRouter
from ragproject.core.chat.adapters.store_memory import InMemoryConversationStore
from ragproject.core.chat.models import (
    ChatPolicy,
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
    RelevanceFilter,
    RetrievalPort,
    RetrievalRouter,
)
from ragproject.core.chat.service import ChatService, ConversationNotFound

__all__ = [
    "AlwaysRetrieveRouter",
    "ChatPolicy",
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
    "RelevanceFilter",
    "RetrievalPort",
    "RetrievalRouter",
    "RouteDecision",
    "ThresholdFilter",
    "StreamEnd",
    "StreamEvent",
    "StreamStart",
    "StreamStatus",
    "StreamToken",
    "Turn",
]
