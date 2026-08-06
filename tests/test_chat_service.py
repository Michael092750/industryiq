"""Unit tests for ChatService -- pure conversation orchestration.

The retrieve job now lives behind a :class:`ContextRetriever`, so these tests
drive ChatService with a fake that returns a canned :class:`RetrievalResult` --
the full multi-turn flow runs with no database and no network. Retrieval behavior
(rewrite, fan-out, filter, merge) is covered separately in
``test_retrieval_service.py``.
"""

import pytest

from industryiq.core.chat.adapters.routing import AlwaysRetrieveRouter
from industryiq.core.chat.adapters.store_memory import InMemoryConversationStore
from industryiq.core.chat.models import (
    ChatPolicy,
    RouteDecision,
    StreamEnd,
    StreamStart,
    StreamStatus,
    StreamToken,
    Turn,
)
from industryiq.core.chat.ports import ConversationStore, TurnRouter
from industryiq.core.chat.service import ChatService, ConversationNotFound
from industryiq.core.generation import FakeLLM, StreamingLLM
from industryiq.core.grounding import DEFAULT_ABSTENTION, DeterministicGroundingGate
from industryiq.core.retrieval.ports import ContextRetriever, RetrievalResult, SessionDocumentStore
from industryiq.core.vectorstore import Hit, MetadataFilter, SearchPlan


class StubRetrieval:
    """A ContextRetriever double: returns canned hits and records gather calls.

    Models the broadening path too: the first (routed) ``gather`` returns ``hits``
    with ``search_plan``; a follow-up call with a ``plan_override`` returns
    ``broadened_hits`` instead (and echoes the override as the result's plan).
    """

    def __init__(
        self,
        hits: list[Hit] | None = None,
        standalone: str = "STANDALONE",
        timings: dict[str, float] | None = None,
        search_plan: SearchPlan | None = None,
        broadened_hits: list[Hit] | None = None,
    ) -> None:
        self._hits = hits or []
        self._standalone = standalone
        self._timings = timings if timings is not None else {"rewrite": 0.0, "retrieve": 0.0}
        self._search_plan = search_plan if search_plan is not None else SearchPlan()
        self._broadened_hits = broadened_hits or []
        self.calls: list[tuple[str, str, list[Turn], int]] = []
        self.plan_overrides: list[SearchPlan | None] = []

    def gather(
        self,
        conversation_id: str,
        question: str,
        history: list[Turn],
        k: int,
        *,
        plan_override: SearchPlan | None = None,
    ) -> RetrievalResult:
        self.calls.append((conversation_id, question, list(history), k))
        self.plan_overrides.append(plan_override)
        if plan_override is not None:
            return RetrievalResult(
                standalone_question=self._standalone,
                hits=list(self._broadened_hits),
                timings_ms=dict(self._timings),
                search_plan=plan_override,
            )
        return RetrievalResult(
            standalone_question=self._standalone,
            hits=list(self._hits),
            timings_ms=dict(self._timings),
            search_plan=self._search_plan,
        )


class StubRouter:
    """A TurnRouter double with a fixed verdict that records its calls."""

    def __init__(self, should_retrieve: bool, needs_planning: bool = False) -> None:
        self._should = should_retrieve
        self._plan = needs_planning
        self.calls: list[tuple[list[Turn], str]] = []

    def route(self, history: list[Turn], question: str) -> RouteDecision:
        self.calls.append((history, question))
        return RouteDecision(should_retrieve=self._should, needs_planning=self._plan)


class FakeOrchestrator:
    """A TurnOrchestrator double: yields canned events (no StreamEnd), records calls."""

    def __init__(self, tokens: tuple[str, ...] = ("plan", "ned")) -> None:
        self._tokens = tokens
        self.calls: list[tuple[list[Turn], str]] = []

    def run_stream(self, history: list[Turn], question: str) -> object:
        self.calls.append((list(history), question))
        yield StreamStatus(phase="planning")
        yield StreamStart(standalone_question="STANDALONE", hits=[])
        for token in self._tokens:
            yield StreamToken(text=token)


class RecordingSessionDocuments:
    """A SessionDocumentStore double that records ``clear`` (lifecycle) calls."""

    def __init__(self) -> None:
        self.cleared: list[str] = []

    def add(self, conversation_id: str, filename: str, text: str) -> list[str]:
        return []

    def retrieve(self, conversation_id: str, query: str, k: int = 5) -> list[Hit]:
        return []

    def documents(self, conversation_id: str) -> list[str]:
        return []

    def clear(self, conversation_id: str) -> None:
        self.cleared.append(conversation_id)


def _service(
    retrieval: ContextRetriever | None = None,
    router: TurnRouter | None = None,
    llm: StreamingLLM | None = None,
    store: ConversationStore | None = None,
    session_documents: SessionDocumentStore | None = None,
    policy: ChatPolicy | None = None,
    orchestrator: object | None = None,
    grounding: object | None = None,
) -> ChatService:
    return ChatService(
        retrieval=retrieval or StubRetrieval(),
        router=router or AlwaysRetrieveRouter(),
        llm=llm or FakeLLM(response="ANSWER"),
        store=store or InMemoryConversationStore(),
        orchestrator=orchestrator,  # duck-typed TurnOrchestrator double
        grounding=grounding,  # duck-typed GroundingGate double (None => gate off)
        session_documents=session_documents,
        policy=policy or ChatPolicy(),
    )


def test_reply_persists_the_turn() -> None:
    store = InMemoryConversationStore()
    service = _service(store=store)
    convo = service.start("c")
    service.reply(convo.id, "hello?")
    assert store.history(convo.id) == [Turn("hello?", "ANSWER")]


def test_reply_delegates_to_retrieval_and_surfaces_the_standalone_question() -> None:
    retrieval = StubRetrieval(standalone="standalone query")
    service = _service(retrieval=retrieval)
    convo = service.start("c")
    result = service.reply(convo.id, "follow up?")
    assert retrieval.calls[0][1] == "follow up?"  # raw question handed to retrieval
    assert result.standalone_question == "standalone query"


def test_reply_grounds_the_prompt_in_retrieved_context() -> None:
    llm = FakeLLM(response="grounded")
    retrieval = StubRetrieval(hits=[Hit("c1", 0.9, {"text": "the sky is blue"})])
    service = _service(retrieval=retrieval, llm=llm)
    convo = service.start("c")
    service.reply(convo.id, "the sky is blue")
    assert llm.last_prompt is not None
    assert "the sky is blue" in llm.last_prompt


def test_reply_to_unknown_conversation_raises() -> None:
    service = _service()
    with pytest.raises(ConversationNotFound):
        service.reply("does-not-exist", "hi")


def test_reply_reports_per_step_timings(fake_clock) -> None:
    store = InMemoryConversationStore()
    service = ChatService(
        retrieval=StubRetrieval(),
        router=AlwaysRetrieveRouter(),
        llm=FakeLLM(),
        store=store,
        clock=fake_clock(step=0.001),
    )
    convo = service.start("c")
    result = service.reply(convo.id, "q")
    # "rewrite"/"retrieve" come from the retrieval result; the rest are chat's own.
    expected = {"load", "route", "rewrite", "retrieve", "generate", "persist", "total"}
    assert expected <= set(result.timings_ms)
    assert all(value >= 0 for value in result.timings_ms.values())


def test_router_decline_skips_retrieval() -> None:
    retrieval = StubRetrieval(hits=[Hit("c1", 0.9, {"text": "x"})])
    service = _service(
        retrieval=retrieval,
        router=StubRouter(should_retrieve=False),
        llm=FakeLLM(response="hi there"),
    )
    convo = service.start("c")
    events = list(service.reply_stream(convo.id, "hello"))
    assert retrieval.calls == []  # never gathered
    start = next(event for event in events if isinstance(event, StreamStart))
    assert start.hits == []
    phases = [event.phase for event in events if isinstance(event, StreamStatus)]
    assert "retrieving" not in phases


def test_reply_surfaces_retrieved_hits_as_sources() -> None:
    retrieval = StubRetrieval(hits=[Hit("c1", 0.9, {"text": "ctx", "source": "d.txt"})])
    service = _service(retrieval=retrieval)
    convo = service.start("c")
    events = list(service.reply_stream(convo.id, "q"))
    start = next(event for event in events if isinstance(event, StreamStart))
    assert [hit.id for hit in start.hits] == ["c1"]
    assert start.hits[0].metadata["source"] == "d.txt"


def test_reply_stream_emits_status_phases() -> None:
    service = _service(llm=FakeLLM(response="hello world"))
    convo = service.start("c")
    events = list(service.reply_stream(convo.id, "q"))
    phases = [event.phase for event in events if isinstance(event, StreamStatus)]
    assert phases == ["thinking", "retrieving", "generating"]


def test_reply_stream_broadens_when_a_filter_yields_no_hits() -> None:
    # A filtered plan returns nothing; ChatService should announce "broadening" and
    # retry once without the filter, then generate from the recovered hits.
    stub = StubRetrieval(
        hits=[],
        search_plan=SearchPlan(filter=MetadataFilter(publisher="McKinsey")),
        broadened_hits=[Hit("c", 0.9, {"text": "grounding"})],
    )
    store = InMemoryConversationStore()
    convo = store.create("c")
    events = list(_service(retrieval=stub, store=store).reply_stream(convo.id, "q"))

    phases = [event.phase for event in events if isinstance(event, StreamStatus)]
    assert phases == ["thinking", "retrieving", "broadening", "generating"]
    # Retried exactly once, the second time with the filter dropped.
    assert len(stub.calls) == 2
    assert stub.plan_overrides[0] is None
    assert stub.plan_overrides[1] is not None and not stub.plan_overrides[1].has_active_filter()
    # The broadened hits reach the answer (grounding recovered, not answered empty).
    start = next(event for event in events if isinstance(event, StreamStart))
    assert [hit.id for hit in start.hits] == ["c"]


def test_reply_stream_does_not_broaden_when_filtered_search_has_hits() -> None:
    # A filter that DOES match must not trigger a retry.
    stub = StubRetrieval(
        hits=[Hit("c", 0.9, {"text": "ctx"})],
        search_plan=SearchPlan(filter=MetadataFilter(publisher="McKinsey")),
    )
    store = InMemoryConversationStore()
    convo = store.create("c")
    events = list(_service(retrieval=stub, store=store).reply_stream(convo.id, "q"))
    phases = [event.phase for event in events if isinstance(event, StreamStatus)]
    assert "broadening" not in phases
    assert len(stub.calls) == 1


def test_reply_stream_does_not_broaden_when_no_filter_was_applied() -> None:
    # Empty hits but no filter (nothing to drop) -> answer empty, no retry.
    stub = StubRetrieval(hits=[], search_plan=SearchPlan())
    store = InMemoryConversationStore()
    convo = store.create("c")
    events = list(_service(retrieval=stub, store=store).reply_stream(convo.id, "q"))
    phases = [event.phase for event in events if isinstance(event, StreamStatus)]
    assert "broadening" not in phases
    assert len(stub.calls) == 1


def test_reply_stream_emits_start_then_tokens_then_end(fake_clock) -> None:
    store = InMemoryConversationStore()
    service = ChatService(
        retrieval=StubRetrieval(hits=[Hit("c1", 0.9, {"text": "ctx", "source": "d.txt"})]),
        router=AlwaysRetrieveRouter(),
        llm=FakeLLM(response="hello world"),
        store=store,
        clock=fake_clock(step=0.001),
    )
    convo = service.start("c")
    events = list(service.reply_stream(convo.id, "q"))

    start = next(event for event in events if isinstance(event, StreamStart))
    assert start.hits[0].metadata["source"] == "d.txt"
    tokens = [event.text for event in events if isinstance(event, StreamToken)]
    assert "".join(tokens) == "hello world"
    assert isinstance(events[-1], StreamEnd)
    assert events[-1].answer == "hello world"
    assert "first_token" in events[-1].timings_ms


def test_reply_stream_persists_full_answer_after_streaming() -> None:
    store = InMemoryConversationStore()
    service = _service(llm=FakeLLM(response="hello world"), store=store)
    convo = service.start("c")
    list(service.reply_stream(convo.id, "q"))  # drain the stream
    assert store.history(convo.id) == [Turn("q", "hello world")]


# --- complex tier: delegation to the agent orchestrator ---------------------------


def test_complex_turn_delegates_to_the_orchestrator_and_persists() -> None:
    store = InMemoryConversationStore()
    convo = store.create("c")
    orchestrator = FakeOrchestrator(tokens=("plan", "ned"))
    retrieval = StubRetrieval(hits=[Hit("c1", 0.9, {"text": "x"})])
    service = _service(
        retrieval=retrieval,
        router=StubRouter(should_retrieve=True, needs_planning=True),
        store=store,
        orchestrator=orchestrator,
    )
    events = list(service.reply_stream(convo.id, "compare A and B"))

    assert orchestrator.calls and orchestrator.calls[0][1] == "compare A and B"
    assert retrieval.calls == []  # the simple retrieve path was NOT taken
    # ChatService forwarded the orchestrator's events...
    assert "planning" in [e.phase for e in events if isinstance(e, StreamStatus)]
    # ...accumulated the tokens, owned the final StreamEnd, and persisted one turn.
    assert isinstance(events[-1], StreamEnd)
    assert events[-1].answer == "planned"
    assert store.history(convo.id) == [Turn("compare A and B", "planned")]


def test_complex_turn_falls_back_to_retrieve_when_no_orchestrator() -> None:
    store = InMemoryConversationStore()
    convo = store.create("c")
    retrieval = StubRetrieval(hits=[Hit("c1", 0.9, {"text": "x"})])
    service = _service(
        retrieval=retrieval,
        router=StubRouter(should_retrieve=True, needs_planning=True),
        store=store,  # no orchestrator wired
    )
    events = list(service.reply_stream(convo.id, "compare A and B"))
    assert retrieval.calls  # degraded safely to the retrieve/simple path
    assert "planning" not in [e.phase for e in events if isinstance(e, StreamStatus)]


# --- grounding gate (retrieve tier) -----------------------------------------------


def test_grounding_gate_abstains_when_retrieval_finds_nothing() -> None:
    # No grounded context -> abstain deterministically instead of letting the model
    # answer (and possibly hallucinate); the generator is never even called.
    llm = FakeLLM(response="hallucinated answer")
    store = InMemoryConversationStore()
    convo = store.create("c")
    service = _service(
        retrieval=StubRetrieval(hits=[]),
        llm=llm,
        store=store,
        grounding=DeterministicGroundingGate(),
    )
    events = list(service.reply_stream(convo.id, "q"))
    assert isinstance(events[-1], StreamEnd)
    assert events[-1].answer == DEFAULT_ABSTENTION
    assert llm.last_prompt is None  # never generated
    assert store.history(convo.id) == [Turn("q", DEFAULT_ABSTENTION)]


def test_grounding_gate_caveats_a_fabricated_citation() -> None:
    # One hit is numbered [1]; the model cites [2], which matches no source -> caveat.
    store = InMemoryConversationStore()
    convo = store.create("c")
    service = _service(
        retrieval=StubRetrieval(hits=[Hit("c1", 0.9, {"text": "ctx"})]),
        llm=FakeLLM(response="the answer [2]"),
        store=store,
        grounding=DeterministicGroundingGate(),
    )
    events = list(service.reply_stream(convo.id, "q"))
    answer = events[-1].answer
    assert answer.startswith("the answer [2]")
    assert "could not be matched" in answer  # the caveat was appended
    assert store.history(convo.id)[0].answer == answer  # persisted with the caveat


def test_grounding_gate_leaves_a_clean_grounded_answer_untouched() -> None:
    service = _service(
        retrieval=StubRetrieval(hits=[Hit("c1", 0.9, {"text": "ctx"})]),
        llm=FakeLLM(response="grounded [1]"),
        grounding=DeterministicGroundingGate(),
    )
    convo = service.start("c")
    assert service.reply(convo.id, "q").answer == "grounded [1]"


def test_grounding_gate_does_not_abstain_on_a_greeting() -> None:
    # should_retrieve=False (small talk) has no context by design -> must NOT abstain.
    service = _service(
        retrieval=StubRetrieval(hits=[]),
        router=StubRouter(should_retrieve=False),
        llm=FakeLLM(response="hello!"),
        grounding=DeterministicGroundingGate(),
    )
    convo = service.start("c")
    assert service.reply(convo.id, "hi").answer == "hello!"


def test_reply_stream_to_unknown_conversation_raises() -> None:
    service = _service()
    with pytest.raises(ConversationNotFound):
        list(service.reply_stream("does-not-exist", "hi"))


def test_list_conversations_returns_all_started() -> None:
    store = InMemoryConversationStore()
    service = _service(store=store)
    a = service.start("a")
    b = service.start("b")
    assert {c.id for c in service.list_conversations()} == {a.id, b.id}


def test_delete_conversation_clears_session_documents() -> None:
    # Session-document lifecycle (clear on delete) is chat's job, not retrieval's.
    docs = RecordingSessionDocuments()
    store = InMemoryConversationStore()
    service = _service(store=store, session_documents=docs)
    convo = service.start("c")
    service.delete_conversation(convo.id)
    assert docs.cleared == [convo.id]


def test_history_limit_caps_history_sent_to_retrieval() -> None:
    retrieval = StubRetrieval()
    store = InMemoryConversationStore()
    service = _service(retrieval=retrieval, store=store, policy=ChatPolicy(history_limit=2))
    convo = service.start("c")
    for i in range(4):
        store.append(convo.id, Turn(f"q{i}", f"a{i}"))
    service.reply(convo.id, "latest")
    _, question, history, _ = retrieval.calls[0]
    assert question == "latest"
    assert history == [Turn("q2", "a2"), Turn("q3", "a3")]
