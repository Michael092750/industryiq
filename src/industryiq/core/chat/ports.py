"""The chat module's ports -- the abstractions it depends on.

Following the Dependency Inversion Principle, :class:`ChatService` is written
against these ``Protocol`` s, never against concrete adapters. Anything that
satisfies a port -- an in-memory fake in a test, Postgres in production -- is
substitutable without touching the service.

Adapters declare their port as an explicit base class (e.g.
``class LlmRouter(TurnRouter)``) so the abstraction-to-implementation link
is visible at the class and mypy verifies it at the definition site.

The *retrieval* seam ChatService depends on -- the coarse
:class:`~industryiq.core.retrieval.ports.ContextRetriever` and the fine-grained
retrieval ports -- lives in :mod:`industryiq.core.retrieval.ports`. Generation
reuses the existing :class:`industryiq.core.generation.LLM` port, so it is not
redefined here.
"""

from collections.abc import Iterator
from typing import Protocol, runtime_checkable

from industryiq.core.chat.models import Conversation, RouteDecision, StreamEvent, Turn


@runtime_checkable
class TurnRouter(Protocol):
    """Decide *which tier* answers a turn: plain reply / retrieve / plan.

    Not merely a retrieve-or-not gate: the returned
    :class:`~industryiq.core.chat.models.RouteDecision` also escalates a complex
    question to the :class:`TurnOrchestrator`, so the port is named for the turn it
    routes rather than for one of the three destinations. The decision drives both
    behavior (skip retrieval for greetings/small talk; plan for a multi-part question)
    and UX (which status the UI shows). Implementations decide how -- always-retrieve,
    an LLM tier classifier, a heuristic.
    """

    def route(self, history: list[Turn], question: str) -> RouteDecision: ...


@runtime_checkable
class TurnOrchestrator(Protocol):
    """Answer a *complex* turn by planning + fanning out to tools, streamed.

    The seam ``ChatService`` delegates to when a turn is
    :attr:`~industryiq.core.chat.models.RouteDecision.needs_planning`. It yields the
    same ``StreamStatus`` / ``StreamStart`` / ``StreamToken`` events the simple path
    does -- but NOT ``StreamEnd``: ``ChatService`` stays the sole owner of the turn
    lifecycle (accumulating the answer, persisting the ``Turn``, emitting the final
    event), so both tiers persist and end a turn identically.

    Implementations live in the ``chat`` package (they bridge to ``agents``); the
    default is :class:`~industryiq.core.chat.adapters.orchestration.AgentTurnOrchestrator`.
    """

    def run_stream(self, history: list[Turn], question: str) -> Iterator[StreamEvent]: ...


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
