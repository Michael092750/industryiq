"""Unit tests for ChatService -- the decoupling showcase.

Every dependency is an in-memory fake or an offline component, so the full
multi-turn flow runs with no database and no network. If the design were
coupled to concrete providers, this file could not exist.
"""

import pytest

from industryiq.core.chat.adapters.filtering import ThresholdFilter
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
from industryiq.core.chat.ports import (
    ConversationStore,
    QueryRewriter,
    RelevanceFilter,
    RetrievalPort,
    RetrievalRouter,
    SessionDocumentStore,
)
from industryiq.core.chat.service import (
    ChatService,
    ConversationNotFound,
    order_session_first,
)
from industryiq.core.embeddings import FakeEmbedder
from industryiq.core.generation import FakeLLM, StreamingLLM
from industryiq.core.retrieval import Retriever
from industryiq.core.vectorstore import Hit, InMemoryVectorStore


class RecordingRewriter:
    """A QueryRewriter double that records calls and returns a canned query."""

    def __init__(self, rewritten: str = "STANDALONE") -> None:
        self._rewritten = rewritten
        self.calls: list[tuple[list[Turn], str]] = []

    def condense(self, history: list[Turn], question: str) -> str:
        self.calls.append((history, question))
        return self._rewritten


class RecordingRetriever:
    """A RetrievalPort double that records the queries it is asked to retrieve."""

    def __init__(self, hits: list[Hit] | None = None) -> None:
        self._hits = hits or []
        self.queries: list[str] = []

    def retrieve(self, query: str, k: int = 5) -> list[Hit]:
        self.queries.append(query)
        return self._hits


class StubRouter:
    """A RetrievalRouter double with a fixed verdict that records its calls."""

    def __init__(self, should_retrieve: bool) -> None:
        self._should = should_retrieve
        self.calls: list[tuple[list[Turn], str]] = []

    def route(self, history: list[Turn], question: str) -> RouteDecision:
        self.calls.append((history, question))
        return RouteDecision(should_retrieve=self._should)


class StubSessionDocuments:
    """A SessionDocumentStore double returning fixed per-session hits."""

    def __init__(self, hits: list[Hit] | None = None) -> None:
        self._hits = hits or []

    def add(self, conversation_id: str, filename: str, text: str) -> list[str]:
        return []

    def retrieve(self, conversation_id: str, query: str, k: int = 5) -> list[Hit]:
        return self._hits

    def documents(self, conversation_id: str) -> list[str]:
        return []


def _service(
    retriever: RetrievalPort | None = None,
    router: RetrievalRouter | None = None,
    rewriter: QueryRewriter | None = None,
    llm: StreamingLLM | None = None,
    store: ConversationStore | None = None,
    relevance_filter: RelevanceFilter | None = None,
    session_documents: SessionDocumentStore | None = None,
) -> ChatService:
    return ChatService(
        retriever=retriever or RecordingRetriever(),
        router=router or AlwaysRetrieveRouter(),
        rewriter=rewriter or RecordingRewriter(),
        llm=llm or FakeLLM(response="ANSWER"),
        store=store or InMemoryConversationStore(),
        relevance_filter=relevance_filter or ThresholdFilter(),
        session_documents=session_documents,
    )


def test_reply_persists_the_turn() -> None:
    store = InMemoryConversationStore()
    service = _service(store=store)
    convo = service.start("c")
    service.reply(convo.id, "hello?")
    assert store.history(convo.id) == [Turn("hello?", "ANSWER")]


def test_reply_retrieves_with_the_rewritten_query() -> None:
    retriever = RecordingRetriever()
    rewriter = RecordingRewriter(rewritten="standalone query")
    service = _service(retriever=retriever, rewriter=rewriter)
    convo = service.start("c")
    result = service.reply(convo.id, "follow up?")
    assert retriever.queries == ["standalone query"]
    assert result.standalone_question == "standalone query"


def test_reply_passes_recent_history_to_the_rewriter() -> None:
    rewriter = RecordingRewriter()
    store = InMemoryConversationStore()
    service = _service(rewriter=rewriter, store=store)
    convo = service.start("c")
    store.append(convo.id, Turn("q1", "a1"))
    service.reply(convo.id, "q2")
    history, question = rewriter.calls[0]
    assert history == [Turn("q1", "a1")]
    assert question == "q2"


def test_reply_grounds_the_prompt_in_retrieved_context() -> None:
    llm = FakeLLM(response="grounded")
    retriever = Retriever(FakeEmbedder(dim=16), InMemoryVectorStore())
    retriever.index(["the sky is blue"])
    service = ChatService(
        retriever=retriever,
        router=AlwaysRetrieveRouter(),
        rewriter=RecordingRewriter(rewritten="the sky is blue"),
        llm=llm,
        store=InMemoryConversationStore(),
        relevance_filter=ThresholdFilter(),
    )
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
        retriever=RecordingRetriever(),
        router=AlwaysRetrieveRouter(),
        rewriter=RecordingRewriter(),
        llm=FakeLLM(),
        store=store,
        relevance_filter=ThresholdFilter(),
        clock=fake_clock(step=0.001),
    )
    convo = service.start("c")
    result = service.reply(convo.id, "q")
    expected = {"load", "route", "rewrite", "retrieve", "generate", "persist", "total"}
    assert expected <= set(result.timings_ms)
    assert all(value >= 0 for value in result.timings_ms.values())


def test_router_decline_skips_retrieval_and_rewrite() -> None:
    retriever = RecordingRetriever(hits=[Hit("c1", 0.9, {"text": "x"})])
    rewriter = RecordingRewriter()
    service = _service(
        retriever=retriever,
        router=StubRouter(should_retrieve=False),
        rewriter=rewriter,
        llm=FakeLLM(response="hi there"),
    )
    convo = service.start("c")
    events = list(service.reply_stream(convo.id, "hello"))
    assert retriever.queries == []  # never searched
    assert rewriter.calls == []  # never condensed
    start = next(event for event in events if isinstance(event, StreamStart))
    assert start.hits == []
    phases = [event.phase for event in events if isinstance(event, StreamStatus)]
    assert "retrieving" not in phases


def test_relevance_filter_drops_hits_below_threshold() -> None:
    retriever = RecordingRetriever(hits=[Hit("c1", 0.2, {"text": "x"})])
    service = _service(retriever=retriever, relevance_filter=ThresholdFilter(threshold=0.5))
    convo = service.start("c")
    events = list(service.reply_stream(convo.id, "q"))
    assert retriever.queries == ["STANDALONE"]  # it DID search
    start = next(event for event in events if isinstance(event, StreamStart))
    assert start.hits == []  # but the low-score hit was filtered out


def test_relevance_filter_keeps_hits_at_or_above_threshold() -> None:
    retriever = RecordingRetriever(hits=[Hit("c1", 0.9, {"text": "x"})])
    service = _service(retriever=retriever, relevance_filter=ThresholdFilter(threshold=0.5))
    convo = service.start("c")
    events = list(service.reply_stream(convo.id, "q"))
    start = next(event for event in events if isinstance(event, StreamStart))
    assert start.hits[0].id == "c1"


def test_reply_stream_emits_status_phases() -> None:
    service = _service(llm=FakeLLM(response="hello world"))
    convo = service.start("c")
    events = list(service.reply_stream(convo.id, "q"))
    phases = [event.phase for event in events if isinstance(event, StreamStatus)]
    assert phases == ["thinking", "retrieving", "generating"]


def test_reply_stream_emits_start_then_tokens_then_end(fake_clock) -> None:
    store = InMemoryConversationStore()
    service = ChatService(
        retriever=RecordingRetriever(hits=[Hit("c1", 0.9, {"text": "ctx", "source": "d.txt"})]),
        router=AlwaysRetrieveRouter(),
        rewriter=RecordingRewriter(),
        llm=FakeLLM(response="hello world"),
        store=store,
        relevance_filter=ThresholdFilter(),
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


def test_order_session_first_leads_with_uploads_over_higher_scoring_shared() -> None:
    session = [Hit("s", 0.2, {})]
    shared = [Hit("g", 0.9, {})]
    # The uploaded session hit leads even though the shared hit scores higher.
    assert [hit.id for hit in order_session_first(session, shared, k=5)] == ["s", "g"]


def test_order_session_first_backfills_remaining_slots_from_shared() -> None:
    session = [Hit("s", 0.4, {})]
    shared = [Hit("g1", 0.9, {}), Hit("g2", 0.1, {})]
    # k=2: one session slot, one shared slot -- the higher-scoring shared hit fills it.
    assert [hit.id for hit in order_session_first(session, shared, k=2)] == ["s", "g1"]


def test_order_session_first_dedups_by_id_keeping_the_session_copy() -> None:
    ordered = order_session_first([Hit("x", 0.4, {})], [Hit("x", 0.8, {})], k=5)
    assert [hit.id for hit in ordered] == ["x"]
    assert ordered[0].score == 0.4  # the session copy leads; the shared duplicate is dropped


def test_reply_puts_session_uploads_first() -> None:
    shared = RecordingRetriever(hits=[Hit("g", 0.9, {"text": "global"})])
    session = StubSessionDocuments(hits=[Hit("s", 0.2, {"text": "session"})])
    service = _service(retriever=shared, session_documents=session)
    convo = service.start("c")
    events = list(service.reply_stream(convo.id, "q"))
    start = next(event for event in events if isinstance(event, StreamStart))
    # Upload leads despite a higher-scoring corpus hit; the corpus still backfills.
    assert [hit.id for hit in start.hits] == ["s", "g"]
    assert shared.queries == ["STANDALONE"]  # corpus consulted to fill the leftover slots


def test_reply_skips_shared_when_session_fills_the_budget() -> None:
    shared = RecordingRetriever(hits=[Hit("g", 0.9, {"text": "global"})])
    session = StubSessionDocuments(hits=[Hit("s", 0.5, {"text": "session"})])
    service = ChatService(
        retriever=shared,
        router=AlwaysRetrieveRouter(),
        rewriter=RecordingRewriter(),
        llm=FakeLLM(response="ANSWER"),
        store=InMemoryConversationStore(),
        relevance_filter=ThresholdFilter(),
        session_documents=session,
        policy=ChatPolicy(k=1),
    )
    convo = service.start("c")
    events = list(service.reply_stream(convo.id, "q"))
    start = next(event for event in events if isinstance(event, StreamStart))
    assert [hit.id for hit in start.hits] == ["s"]  # only the upload
    assert shared.queries == []  # corpus never consulted -- the upload already filled k


def test_history_limit_caps_turns_sent_to_the_rewriter() -> None:
    rewriter = RecordingRewriter()
    store = InMemoryConversationStore()
    service = ChatService(
        retriever=RecordingRetriever(),
        router=AlwaysRetrieveRouter(),
        rewriter=rewriter,
        llm=FakeLLM(),
        store=store,
        relevance_filter=ThresholdFilter(),
        policy=ChatPolicy(history_limit=2),
    )
    convo = service.start("c")
    for i in range(4):
        store.append(convo.id, Turn(f"q{i}", f"a{i}"))
    service.reply(convo.id, "latest")
    history, _ = rewriter.calls[0]
    assert history == [Turn("q2", "a2"), Turn("q3", "a3")]
