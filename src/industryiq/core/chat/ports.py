"""The chat module's ports -- the abstractions it depends on.

Following the Dependency Inversion Principle, :class:`ChatService` is written
against these ``Protocol`` s, never against concrete adapters. Anything that
satisfies a port -- an in-memory fake in a test, Postgres in production -- is
substitutable without touching the service.

Adapters declare their port as an explicit base class (e.g.
``class LlmRouter(RetrievalRouter)``) so the abstraction-to-implementation link
is visible at the class and mypy verifies it at the definition site.

The *retrieval* seam ChatService depends on -- the coarse
:class:`~industryiq.core.retrieval.ports.ContextRetriever` and the fine-grained
retrieval ports -- lives in :mod:`industryiq.core.retrieval.ports`. Generation
reuses the existing :class:`industryiq.core.generation.LLM` port, so it is not
redefined here.
"""

from typing import Protocol, runtime_checkable

from industryiq.core.chat.models import Conversation, RouteDecision, Turn


@runtime_checkable
class RetrievalRouter(Protocol):
    """Decide whether answering a question needs a knowledge-base lookup.

    The decision drives both behavior (skip retrieval for greetings/small talk)
    and UX (whether to show a "checking knowledge base" status). Implementations
    decide how -- always, an LLM intent classifier, a heuristic.
    """

    def route(self, history: list[Turn], question: str) -> RouteDecision: ...


@runtime_checkable
class ConversationStore(Protocol):
    """Persist conversations and their turns.

    ``owner_id`` scopes ownership: ``create`` records it, and ``list_all`` filters
    by it. ``None`` means "no scoping" -- every conversation -- which the default
    (pre-auth) and test paths rely on; the API always passes a real user.
    """

    def create(self, title: str, owner_id: str | None = None) -> Conversation: ...

    def get(self, conversation_id: str) -> Conversation | None: ...

    def history(self, conversation_id: str, limit: int | None = None) -> list[Turn]: ...

    def append(self, conversation_id: str, turn: Turn) -> None: ...

    def rename(self, conversation_id: str, title: str) -> None: ...

    def delete(self, conversation_id: str) -> None: ...

    def list_all(self, owner_id: str | None = None) -> list[Conversation]: ...
