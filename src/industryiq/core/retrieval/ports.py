"""The retrieval package's ports -- the abstractions the retrieval side depends on.

Following the Dependency Inversion Principle, :class:`RetrievalService` is written
against these ``Protocol``s, never against concrete adapters. They were promoted
out of the chat package so retrieval is self-contained and a supervisor/agent can
compose it without pulling in ``chat``.

* :class:`RetrievalPort`, :class:`RelevanceFilter`, :class:`QueryRewriter`,
  :class:`SessionDocumentStore` -- the fine-grained collaborators the service
  composes.
* :class:`ContextRetriever` + :class:`RetrievalResult` -- the *coarse* seam a
  caller (e.g. :class:`~industryiq.core.chat.service.ChatService`) depends on:
  "gather the grounding context for this turn" in, a filtered/merged result out.
"""

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from industryiq.core.conversation import Turn
from industryiq.core.vectorstore import Hit


@runtime_checkable
class RetrievalPort(Protocol):
    """Find the chunks most relevant to a query.

    Deliberately narrow (Interface Segregation): a retrieval consumer only ever
    *reads*, so it depends on ``retrieve`` alone, not on indexing. The concrete
    :class:`~industryiq.core.retrieval.retriever.Retriever` satisfies this
    structurally.
    """

    def retrieve(self, query: str, k: int = 5) -> list[Hit]: ...


@runtime_checkable
class RelevanceFilter(Protocol):
    """Decide which retrieved hits are relevant enough to ground the answer.

    The post-retrieval coverage gate ("did we actually find anything useful?"),
    symmetric with the pre-retrieval routing decision. Implementations decide how
    -- a score threshold, a reranker, a quorum rule.
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
class SessionDocumentStore(Protocol):
    """Ephemeral per-conversation document index (in memory, lost on restart).

    Lets a chat search the files uploaded into *that session* only, separate
    from the persistent shared knowledge base. Uploaded documents are the
    primary context: they lead the results, and the shared store only backfills
    (see :func:`industryiq.core.retrieval.service.order_session_first`). Same
    embedder as the shared store, so scores stay comparable within each
    backfilled slot.

    The index lives here (a retrieval construct), but its lifecycle methods
    (``add`` / ``documents`` / ``clear``) are driven by chat -- upload/list
    routes and conversation deletion -- while ``retrieve`` is used by
    :class:`RetrievalService`.
    """

    def add(self, conversation_id: str, filename: str, text: str) -> list[str]: ...

    def retrieve(self, conversation_id: str, query: str, k: int = 5) -> list[Hit]: ...

    def documents(self, conversation_id: str) -> list[str]: ...

    def clear(self, conversation_id: str) -> None: ...


@dataclass(frozen=True)
class RetrievalResult:
    """The grounding context gathered for one turn.

    * ``standalone_question`` -- the condensed, self-contained query actually used
      for retrieval (surfaced for debugging follow-ups).
    * ``hits`` -- the filtered, merged chunks to ground the answer on.
    * ``timings_ms`` -- per-step durations (``"rewrite"``, ``"retrieve"``) so the
      caller can fold them into its own turn timings.
    """

    standalone_question: str
    hits: list[Hit]
    timings_ms: dict[str, float]


@runtime_checkable
class ContextRetriever(Protocol):
    """Gather the grounding context for one conversational turn.

    The coarse retrieval seam: a caller hands over the raw question plus recent
    history and gets back a :class:`RetrievalResult` (rewrite + all doc retrieval
    + relevance filtering + merge already applied). Implementations decide the
    strategy; :class:`RetrievalService` is the default.
    """

    def gather(
        self, conversation_id: str, question: str, history: list[Turn], k: int
    ) -> RetrievalResult: ...
