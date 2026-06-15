"""The chat module's ports -- the abstractions it depends on.

Following the Dependency Inversion Principle, :class:`ChatService` is written
against these ``Protocol`` s, never against concrete adapters. Anything that
satisfies a port -- an in-memory fake in a test, Postgres in production -- is
substitutable without touching the service.

Generation reuses the existing :class:`ragproject.core.generation.LLM` port, so
it is not redefined here.
"""

from typing import Protocol, runtime_checkable

from ragproject.core.chat.models import Conversation, RouteDecision, Turn
from ragproject.core.vectorstore import Hit


@runtime_checkable
class RetrievalPort(Protocol):
    """Find the chunks most relevant to a query.

    Deliberately narrower than :class:`ragproject.core.retrieval.Retriever`
    (Interface Segregation): chat only ever *reads*, so it depends on
    ``retrieve`` alone, not on indexing. The existing ``Retriever`` satisfies
    this structurally.
    """

    def retrieve(self, query: str, k: int = 5) -> list[Hit]: ...


@runtime_checkable
class RetrievalRouter(Protocol):
    """Decide whether answering a question needs a knowledge-base lookup.

    The decision drives both behavior (skip retrieval for greetings/small talk)
    and UX (whether to show a "checking knowledge base" status). Implementations
    decide how -- always, an LLM intent classifier, a heuristic.
    """

    def route(self, history: list[Turn], question: str) -> RouteDecision: ...


@runtime_checkable
class QueryRewriter(Protocol):
    """Rewrite a follow-up question into a standalone one.

    "What about its pricing?" only makes sense given prior turns, but retrieval
    needs a self-contained query. Implementations decide how (LLM, no-op, ...).
    """

    def condense(self, history: list[Turn], question: str) -> str: ...


@runtime_checkable
class ConversationStore(Protocol):
    """Persist conversations and their turns."""

    def create(self, title: str) -> Conversation: ...

    def get(self, conversation_id: str) -> Conversation | None: ...

    def history(self, conversation_id: str, limit: int | None = None) -> list[Turn]: ...

    def append(self, conversation_id: str, turn: Turn) -> None: ...
