"""ChatService: the orchestration policy for one conversational turn.

High-level policy that depends only on ports (Dependency Inversion). It holds no
SQL and makes no provider calls of its own -- it *coordinates* the router, the
retrieval service, the LLM, and the store. That is the whole reason it can be
unit tested end to end with in-memory fakes and zero network.

Per turn it: routes (which tier answers this -- plain reply, retrieve, or plan?),
and on the retrieve tier delegates the whole retrieve job to a
:class:`~industryiq.core.retrieval.ports.ContextRetriever` (rewrite + fan-out +
filter + merge), then generates; a planning turn goes to the
:class:`~industryiq.core.chat.ports.TurnOrchestrator`. :meth:`reply_stream` yields
the answer token by token with status events for the UI; :meth:`reply` simply
drains that stream into a :class:`ChatResult`, so the two can never diverge.

Each phase is timed (see :class:`StepTimer`).
"""

import time
from collections.abc import Callable, Generator, Iterator

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
from industryiq.core.chat.prompting import build_chat_prompt
from industryiq.core.generation import StreamingLLM
from industryiq.core.grounding import (
    DEFAULT_ABSTENTION,
    GroundingGate,
    citation_caveat,
    verify_citations,
)
from industryiq.core.retrieval.ports import ContextRetriever, SessionDocumentStore
from industryiq.core.timing import StepTimer
from industryiq.core.vectorstore import Hit


class ConversationNotFound(Exception):
    """Raised when an operation targets a conversation that does not exist."""


# ChatPolicy is immutable, so one shared default instance is safe to reuse.
_DEFAULT_POLICY = ChatPolicy()


class ChatService:
    """Coordinate routing, retrieval, generation, and persistence for a turn.

    Pure conversation orchestration: it routes the turn to a tier, delegates the
    whole retrieve job to a :class:`ContextRetriever` (or the turn itself to the
    :class:`TurnOrchestrator`), then generates and persists. It owns conversation
    *lifecycle* -- history, and the session-document ``clear`` on delete -- but no
    retrieval policy of its own.
    """

    def __init__(
        self,
        retrieval: ContextRetriever,
        router: TurnRouter,
        llm: StreamingLLM,
        store: ConversationStore,
        *,
        orchestrator: TurnOrchestrator | None = None,
        grounding: GroundingGate | None = None,
        session_documents: SessionDocumentStore | None = None,
        policy: ChatPolicy = _DEFAULT_POLICY,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self._retrieval = retrieval
        self._router = router
        self._llm = llm
        self._store = store
        # The complex-turn tier: when a turn is routed ``needs_planning`` and an
        # orchestrator is wired, the whole answer is delegated to it (plan -> fan-out
        # -> streamed synthesis). ``None`` => complex turns fall back to the simple
        # retrieve path, so behaviour degrades safely (e.g. offline).
        self._orchestrator = orchestrator
        # Post-generation faithfulness check for the retrieve tier: abstain when there
        # is no grounded context, and caveat fabricated citations. ``None`` => the gate
        # is off (today's behaviour), so existing/offline callers are unaffected.
        self._grounding = grounding
        # Held only for lifecycle: clearing a conversation's uploads on delete.
        # Retrieval *from* it is the retrieval service's job, not chat's.
        self._session_documents = session_documents
        self._policy = policy
        self._clock = clock

    def start(self, title: str, owner_id: str | None = None) -> Conversation:
        """Open a new conversation, owned by ``owner_id`` when given."""
        return self._store.create(title, owner_id)

    def get_conversation(self, conversation_id: str) -> Conversation | None:
        """Return the conversation, or ``None`` if it does not exist."""
        return self._store.get(conversation_id)

    def get_history(self, conversation_id: str) -> list[Turn]:
        """Return the full turn history of a conversation (for display)."""
        return self._store.history(conversation_id)

    def list_conversations(self, owner_id: str | None = None) -> list[Conversation]:
        """Return conversations (optionally only ``owner_id``'s), newest first."""
        return self._store.list_all(owner_id)

    def rename_conversation(self, conversation_id: str, title: str) -> None:
        """Change a conversation's title."""
        self._store.rename(conversation_id, title)

    def delete_conversation(self, conversation_id: str) -> None:
        """Delete a conversation, its turns, and any documents uploaded into it."""
        self._store.delete(conversation_id)
        if self._session_documents is not None:
            self._session_documents.clear(conversation_id)

    def reply(self, conversation_id: str, question: str) -> ChatResult:
        """Answer ``question``, returning the complete result (drains the stream)."""
        hits: list[Hit] = []
        standalone = question
        answer = ""
        timings: dict[str, float] = {}
        for event in self.reply_stream(conversation_id, question):
            if isinstance(event, StreamStart):
                standalone, hits = event.standalone_question, event.hits
            elif isinstance(event, StreamEnd):
                answer, timings = event.answer, event.timings_ms
        return ChatResult(
            answer=answer,
            hits=hits,
            standalone_question=standalone,
            timings_ms=timings,
        )

    def reply_stream(self, conversation_id: str, question: str) -> Iterator[StreamEvent]:
        """Answer ``question`` incrementally.

        Yields ``StreamStatus`` phase markers, a ``StreamStart`` (sources, possibly
        empty), a ``StreamToken`` per chunk, and a final ``StreamEnd``. The turn is
        persisted just before the final event.
        """
        timer = StepTimer(self._clock)
        with timer.measure("total"):
            yield StreamStatus(phase="thinking")

            with timer.measure("load"):
                if self._store.get(conversation_id) is None:
                    raise ConversationNotFound(conversation_id)
                history = self._store.history(conversation_id, limit=self._policy.history_limit)

            with timer.measure("route"):
                decision = self._router.route(history, question)

            # Complex tier: delegate the whole answer to the agent orchestrator
            # (plan -> fan-out -> streamed synthesis), forwarding its events. Falls
            # back to the retrieve/simple path when no orchestrator is wired.
            if decision.needs_planning and self._orchestrator is not None:
                answer = yield from self._answer_by_planning(history, question, timer)
            else:
                answer = yield from self._answer_from_context(
                    conversation_id, history, question, decision, timer
                )

            with timer.measure("persist"):
                self._store.append(conversation_id, Turn(question=question, answer=answer))
        yield StreamEnd(answer=answer, timings_ms=timer.timings_ms)

    def _answer_from_context(
        self,
        conversation_id: str,
        history: list[Turn],
        question: str,
        decision: RouteDecision,
        timer: StepTimer,
    ) -> Generator[StreamEvent, None, str]:
        """Simple/retrieve tier: (maybe) retrieve, then stream a grounded answer.

        Returns the full answer text (via ``return`` -> the ``yield from`` value).
        """
        standalone = question
        hits: list[Hit] = []
        if decision.should_retrieve:
            yield StreamStatus(phase="retrieving")
            # Delegate the whole retrieve job -- rewrite, fan-out to session + shared
            # sources, relevance filter, merge -- to the retrieval service, and fold
            # its "rewrite"/"retrieve" timings into the turn.
            result = self._retrieval.gather(conversation_id, question, history, self._policy.k)
            timer.timings_ms.update(result.timings_ms)
            # A metadata filter can over-constrain the search to zero hits (e.g. a
            # publisher the corpus doesn't tag that way). Rather than answer with no
            # grounding, tell the UI we're broadening and retry once without the
            # filter (same strategy), then generate from those results.
            if not result.hits and result.search_plan.has_active_filter():
                yield StreamStatus(phase="broadening")
                result = self._retrieval.gather(
                    conversation_id,
                    question,
                    history,
                    self._policy.k,
                    plan_override=result.search_plan.without_filter(),
                )
                timer.timings_ms.update(result.timings_ms)
            standalone, hits = result.standalone_question, result.hits

        yield StreamStart(standalone_question=standalone, hits=hits)
        yield StreamStatus(phase="generating")

        # Grounding gate, pre-generation: with no grounded context, abstain rather than
        # let the model answer from parametric memory (or hallucinate). Retrieve tier
        # only -- greetings (should_retrieve=False) legitimately have no context.
        if self._grounding is not None and decision.should_retrieve and not hits:
            verdict = self._grounding.check(question, "", hits)
            return (yield from self._stream_text(verdict.abstention or DEFAULT_ABSTENTION, timer))

        prompt = build_chat_prompt(history, question, hits)
        parts: list[str] = []
        generate_start = self._clock()
        for chunk in self._llm.stream(prompt):
            if not parts:  # first chunk -> record time-to-first-token
                timer.timings_ms["first_token"] = round((self._clock() - generate_start) * 1000, 3)
            parts.append(chunk)
            yield StreamToken(text=chunk)
        timer.timings_ms["generate"] = round((self._clock() - generate_start) * 1000, 3)
        answer = "".join(parts)

        # Grounding gate, post-generation: keep the streamed answer but append a caveat
        # for any fabricated [n] citation (the tokens are already out, so we suffix
        # rather than replace). Citation validity is a pure fact, so it is checked
        # directly -- no model call on the hot streaming path. A gate verdict that
        # *replaces* the answer needs a surface that can withhold it until the verdict
        # is in, i.e. the non-streamed Synthesizer.synthesize behind /agents/run.
        if self._grounding is not None and hits:
            invalid = verify_citations(answer, hits)
            if invalid:
                caveat = citation_caveat(invalid)
                yield StreamToken(text=caveat)
                answer += caveat
        return answer

    def _stream_text(self, text: str, timer: StepTimer) -> Generator[StreamEvent, None, str]:
        """Stream a fixed answer (e.g. an abstention) as one token, recording timings."""
        start = self._clock()
        timer.timings_ms["first_token"] = round((self._clock() - start) * 1000, 3)
        yield StreamToken(text=text)
        timer.timings_ms["generate"] = round((self._clock() - start) * 1000, 3)
        return text

    def _answer_by_planning(
        self, history: list[Turn], question: str, timer: StepTimer
    ) -> Generator[StreamEvent, None, str]:
        """Complex tier: forward the orchestrator's events, accumulate the answer.

        ChatService stays the turn owner -- it does not emit ``StreamEnd`` here; that
        (and persistence) happen back in :meth:`reply_stream`.
        """
        assert self._orchestrator is not None
        parts: list[str] = []
        start = self._clock()
        for event in self._orchestrator.run_stream(history, question):
            if isinstance(event, StreamToken):
                if not parts:  # first token -> time from delegation start (incl. planning)
                    timer.timings_ms["first_token"] = round((self._clock() - start) * 1000, 3)
                parts.append(event.text)
            yield event
        timer.timings_ms["generate"] = round((self._clock() - start) * 1000, 3)
        return "".join(parts)
