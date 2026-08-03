"""Multi-round chat: a RAG chatbot built as ports-and-adapters.

Public surface:

* :class:`ChatService` -- orchestrates a single conversational turn.
* Ports (:mod:`industryiq.core.chat.ports`) -- the abstractions it depends on.
* Adapters (:mod:`industryiq.core.chat.adapters`) -- the concrete
  implementations of the ports. The Postgres store is imported only where it is
  wired (:mod:`industryiq.api.deps`), to keep this package import light.

The retrieve job lives in its own package,
:mod:`industryiq.core.retrieval` (``RetrievalService`` + its ports/adapters);
import retrieval collaborators from there, not from here.
"""

from industryiq.core.chat.adapters.routing import AlwaysRetrieveRouter, LlmRouter
from industryiq.core.chat.adapters.store_memory import InMemoryConversationStore
from industryiq.core.chat.models import (
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
from industryiq.core.chat.ports import ConversationStore, TurnOrchestrator, TurnRouter
from industryiq.core.chat.service import ChatService, ConversationNotFound

__all__ = [
    "AlwaysRetrieveRouter",
    "ChatPolicy",
    "ChatResult",
    "ChatService",
    "Conversation",
    "ConversationNotFound",
    "ConversationStore",
    "InMemoryConversationStore",
    "LlmRouter",
    "RouteDecision",
    "StreamEnd",
    "StreamEvent",
    "StreamStart",
    "StreamStatus",
    "StreamToken",
    "Turn",
    "TurnOrchestrator",
    "TurnRouter",
]
