"""RetrievalService: the retrieve job as a standalone service.

Owns everything between "we've decided to retrieve" and "here is the grounding
context": condense the follow-up into a standalone query, fan out to the two
sources (session uploads + shared corpus), relevance-filter each, and merge --
returning a :class:`RetrievalResult`. Lifted out of ``ChatService`` so retrieval
is a single unit a supervisor/agent can call, and so the chat turn loop stays
pure orchestration.

It depends only on ports (:mod:`industryiq.core.retrieval.ports`) plus the
neutral conversation + timing helpers -- never on ``chat`` -- keeping the package
dependency one-way.
"""

import time
from collections.abc import Callable

from industryiq.core.conversation import Turn
from industryiq.core.retrieval.ports import (
    ContextExpander,
    ContextRetriever,
    QueryRewriter,
    RelevanceFilter,
    RetrievalPort,
    RetrievalResult,
    SearchStrategyRouter,
    SessionDocumentStore,
)
from industryiq.core.timing import StepTimer
from industryiq.core.vectorstore import Hit, SearchPlan


def order_session_first(session: list[Hit], shared: list[Hit], k: int) -> list[Hit]:
    """Order uploaded session documents ahead of the shared corpus, capped at ``k``.

    Documents uploaded into a conversation are the user's chosen context, so they
    lead: every session hit -- in descending score order -- comes before any
    shared-corpus hit *regardless of score*. The shared knowledge base only
    backfills the slots the uploads leave open. De-duped by id so a chunk somehow
    present in both stores is never listed twice (the session copy wins).
    """
    ordered = sorted(session, key=lambda hit: hit.score, reverse=True)
    seen = {hit.id for hit in ordered}
    for hit in sorted(shared, key=lambda hit: hit.score, reverse=True):
        if len(ordered) >= k:
            break
        if hit.id not in seen:
            ordered.append(hit)
            seen.add(hit.id)
    return ordered[:k]


class RetrievalService(ContextRetriever):
    """Compose rewrite + fan-out + filter + merge into one ``gather`` call."""

    def __init__(
        self,
        retriever: RetrievalPort,
        rewriter: QueryRewriter,
        relevance_filter: RelevanceFilter,
        *,
        strategy_router: SearchStrategyRouter | None = None,
        expander: ContextExpander | None = None,
        session_documents: SessionDocumentStore | None = None,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self._retriever = retriever
        self._rewriter = rewriter
        self._relevance_filter = relevance_filter
        # None => no strategy routing: retrieve with the default plan (hybrid-RRF),
        # i.e. today's behaviour, without coupling the service to an adapter.
        self._strategy_router = strategy_router
        # None => no context expansion: hits are grounded on the matched chunks alone.
        self._expander = expander
        self._session_documents = session_documents
        self._clock = clock

    def gather(
        self, conversation_id: str, question: str, history: list[Turn], k: int
    ) -> RetrievalResult:
        """Rewrite, retrieve from both sources, filter, and merge into a result.
        Records ``"rewrite"`` and ``"retrieve"`` timings so the caller can fold
        them into its own per-turn timings.
        """
        timer = StepTimer(self._clock)
        with timer.measure("rewrite"):
            standalone = self._rewriter.condense(history, question)
        with timer.measure("route_strategy"):
            # Choose *how* to search for the (standalone) question. Session-doc
            # retrieval always stays on the default path (its store isn't
            # strategy-capable); the plan only steers the shared-corpus search.
            plan = (
                self._strategy_router.select(standalone)
                if self._strategy_router is not None
                else SearchPlan()
            )
        with timer.measure("retrieve"):
            # Documents uploaded into this session are the primary context: take
            # (and relevance-filter) them first. Only when they don't fill the
            # whole k-budget do we consult the shared corpus for additional
            # grounding -- "retrieve from the knowledge base if needed". Both go
            # through the same coverage backstop.
            session = self._relevance_filter.keep(
                self._session_documents.retrieve(conversation_id, standalone, k)
                if self._session_documents is not None
                else []
            )
            shared = (
                self._relevance_filter.keep(self._retriever.retrieve(standalone, k=k, plan=plan))
                if len(session) < k
                else []
            )
            hits = order_session_first(session, shared, k)
        if self._expander is not None:
            with timer.measure("expand"):
                # Widen each hit with its neighbouring chunks so a fact straddling a
                # chunk boundary is still in the grounding context.
                hits = self._expander.expand(hits)
        return RetrievalResult(
            standalone_question=standalone,
            hits=hits,
            timings_ms=timer.timings_ms,
            search_plan=plan,
        )
