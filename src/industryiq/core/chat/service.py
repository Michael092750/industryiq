"""ChatService: the orchestration policy for one conversational turn.

High-level policy that depends only on ports (Dependency Inversion). It holds no
SQL, no prompt strings, and makes no provider calls of its own -- it *coordinates*
the router, rewriter, retriever, LLM, and store. That is the whole reason it can
be unit tested end to end with in-memory fakes and zero network.

Per turn it: routes (does this need the knowledge base?), optionally retrieves
and applies a relevance backstop, then generates. :meth:`reply_stream` yields the
answer token by token with status events for the UI; :meth:`reply` simply drains
that stream into a :class:`ChatResult`, so the two can never diverge.

Each phase is timed (see :class:`StepTimer`).
"""

import time
from collections.abc import Callable, Iterator

from industryiq.core.chat.models import (
    ChatPolicy,
    ChatResult,
    Conversation,
    StreamEnd,
    StreamEvent,
    StreamStart,
    StreamStatus,
    StreamToken,
    Turn,
)
from industryiq.core.chat.ports import (
    ConversationStore,
    QueryRewriter,
    RelevanceFilter,
    RetrievalPort,
    RetrievalRouter,
    SessionDocumentStore,
)
from industryiq.core.chat.prompting import build_chat_prompt
from industryiq.core.chat.timing import StepTimer
from industryiq.core.generation import StreamingLLM
from industryiq.core.vectorstore import Hit


class ConversationNotFound(Exception):
    """Raised when an operation targets a conversation that does not exist."""


# ChatPolicy is immutable, so one shared default instance is safe to reuse.
_DEFAULT_POLICY = ChatPolicy()


def merge_hits(primary: list[Hit], secondary: list[Hit], k: int) -> list[Hit]:
    """Combine two hit lists into the top ``k`` by score, de-duplicated by id.

    Both lists come from the same embedder (shared store + session docs), so
    their scores are comparable.
    """
    best: dict[str, Hit] = {}
    for hit in (*primary, *secondary):
        current = best.get(hit.id)
        if current is None or hit.score > current.score:
            best[hit.id] = hit
    return sorted(best.values(), key=lambda hit: hit.score, reverse=True)[:k]


class ChatService:
    """Coordinate routing, rewriting, retrieval, generation, and persistence."""

    def __init__(
        self,
        retriever: RetrievalPort,
        router: RetrievalRouter,
        rewriter: QueryRewriter,
        llm: StreamingLLM,
        store: ConversationStore,
        relevance_filter: RelevanceFilter,
        *,
        session_documents: SessionDocumentStore | None = None,
        policy: ChatPolicy = _DEFAULT_POLICY,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self._retriever = retriever
        self._router = router
        self._rewriter = rewriter
        self._llm = llm
        self._store = store
        self._relevance_filter = relevance_filter
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

            standalone = question
            hits: list[Hit] = []
            if decision.should_retrieve:
                yield StreamStatus(phase="retrieving")
                with timer.measure("rewrite"):
                    standalone = self._rewriter.condense(history, question)
                with timer.measure("retrieve"):
                    shared = self._retriever.retrieve(standalone, k=self._policy.k)
                    session = (
                        self._session_documents.retrieve(
                            conversation_id, standalone, self._policy.k
                        )
                        if self._session_documents is not None
                        else []
                    )
                    retrieved = merge_hits(shared, session, self._policy.k)
                # Coverage backstop: drop hits not relevant enough to ground on.
                hits = self._relevance_filter.keep(retrieved)

            yield StreamStart(standalone_question=standalone, hits=hits)
            yield StreamStatus(phase="generating")

            prompt = build_chat_prompt(history, question, hits)
            parts: list[str] = []
            generate_start = self._clock()
            for chunk in self._llm.stream(prompt):
                if not parts:  # first chunk -> record time-to-first-token
                    timer.timings_ms["first_token"] = round(
                        (self._clock() - generate_start) * 1000, 3
                    )
                parts.append(chunk)
                yield StreamToken(text=chunk)
            timer.timings_ms["generate"] = round((self._clock() - generate_start) * 1000, 3)

            answer = "".join(parts)
            with timer.measure("persist"):
                self._store.append(conversation_id, Turn(question=question, answer=answer))
        yield StreamEnd(answer=answer, timings_ms=timer.timings_ms)
