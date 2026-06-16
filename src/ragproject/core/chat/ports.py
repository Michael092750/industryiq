"""The chat module's ports -- the abstractions it depends on.

Following the Dependency Inversion Principle, :class:`ChatService` is written
against these ``Protocol`` s, never against concrete adapters. Anything that
satisfies a port -- an in-memory fake in a test, Postgres in production -- is
substitutable without touching the service.

Adapters declare their port as an explicit base class (e.g.
``class LlmRouter(RetrievalRouter)``) so the abstraction-to-implementation link
is visible at the class and mypy verifies it at the definition site. The ports
stay ``Protocol`` s rather than ABCs, so explicit subclassing is opt-in:
:class:`RetrievalPort` is the deliberate exception -- left purely structural so
the pre-existing :class:`ragproject.core.retrieval.Retriever`, which lives in
``core`` and must not import ``chat``, can satisfy it without inheritance.

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
class RelevanceFilter(Protocol):
    """Decide which retrieved hits are relevant enough to ground the answer.

    The post-retrieval coverage gate ("did we actually find anything useful?"),
    symmetric with the pre-retrieval :class:`RetrievalRouter`. Implementations
    decide how -- a score threshold, a reranker, a quorum rule.
    """

    def keep(self, hits: list[Hit]) -> list[Hit]: ...


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

    def list_all(self) -> list[Conversation]: ...


@runtime_checkable
class SessionDocumentStore(Protocol):
    """Ephemeral per-conversation document index (in memory, lost on restart).

    Lets a chat search the files uploaded into *that session* only, separate
    from the persistent shared knowledge base. Same embedder as the shared
    store, so the two result sets can be merged by score.
    """

    def add(self, conversation_id: str, filename: str, text: str) -> list[str]: ...

    def retrieve(self, conversation_id: str, query: str, k: int = 5) -> list[Hit]: ...

    def documents(self, conversation_id: str) -> list[str]: ...
