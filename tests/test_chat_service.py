"""Unit tests for ChatService -- the decoupling showcase.

Every dependency is an in-memory fake or an offline component, so the full
multi-turn flow runs with no database and no network. If the design were
coupled to concrete providers, this file could not exist.
"""

import pytest

from ragproject.core.chat.models import (
    RouteDecision,
    StreamEnd,
    StreamStart,
    StreamStatus,
    StreamToken,
    Turn,
)
from ragproject.core.chat.ports import (
    ConversationStore,
    QueryRewriter,
    RetrievalPort,
    RetrievalRouter,
)
from ragproject.core.chat.routing import AlwaysRetrieveRouter
from ragproject.core.chat.service import ChatService, ConversationNotFound
from ragproject.core.chat.store_memory import InMemoryConversationStore
from ragproject.core.embeddings import FakeEmbedder
from ragproject.core.generation import FakeLLM, StreamingLLM
from ragproject.core.retrieval import Retriever
from ragproject.core.vectorstore import Hit, InMemoryVectorStore


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


def _service(
    retriever: RetrievalPort | None = None,
    router: RetrievalRouter | None = None,
    rewriter: QueryRewriter | None = None,
    llm: StreamingLLM | None = None,
    store: ConversationStore | None = None,
) -> ChatService:
    return ChatService(
        retriever=retriever or RecordingRetriever(),
        router=router or AlwaysRetrieveRouter(),
        rewriter=rewriter or RecordingRewriter(),
        llm=llm or FakeLLM(response="ANSWER"),
        store=store or InMemoryConversationStore(),
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


def test_relevance_backstop_drops_hits_below_threshold() -> None:
    retriever = RecordingRetriever(hits=[Hit("c1", 0.2, {"text": "x"})])
    service = ChatService(
        retriever=retriever,
        router=AlwaysRetrieveRouter(),
        rewriter=RecordingRewriter(),
        llm=FakeLLM(),
        store=InMemoryConversationStore(),
        relevance_threshold=0.5,
    )
    convo = service.start("c")
    events = list(service.reply_stream(convo.id, "q"))
    assert retriever.queries == ["STANDALONE"]  # it DID search
    start = next(event for event in events if isinstance(event, StreamStart))
    assert start.hits == []  # but the low-score hit was dropped


def test_relevance_backstop_keeps_hits_at_or_above_threshold() -> None:
    retriever = RecordingRetriever(hits=[Hit("c1", 0.9, {"text": "x"})])
    service = ChatService(
        retriever=retriever,
        router=AlwaysRetrieveRouter(),
        rewriter=RecordingRewriter(),
        llm=FakeLLM(),
        store=InMemoryConversationStore(),
        relevance_threshold=0.5,
    )
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
    label = next(
        event.label
        for event in events
        if isinstance(event, StreamStatus) and event.phase == "retrieving"
    )
    assert "knowledge base" in label


def test_reply_stream_emits_start_then_tokens_then_end(fake_clock) -> None:
    store = InMemoryConversationStore()
    service = ChatService(
        retriever=RecordingRetriever(hits=[Hit("c1", 0.9, {"text": "ctx", "source": "d.txt"})]),
        router=AlwaysRetrieveRouter(),
        rewriter=RecordingRewriter(),
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


def test_reply_stream_to_unknown_conversation_raises() -> None:
    service = _service()
    with pytest.raises(ConversationNotFound):
        list(service.reply_stream("does-not-exist", "hi"))


def test_history_limit_caps_turns_sent_to_the_rewriter() -> None:
    rewriter = RecordingRewriter()
    store = InMemoryConversationStore()
    service = ChatService(
        retriever=RecordingRetriever(),
        router=AlwaysRetrieveRouter(),
        rewriter=rewriter,
        llm=FakeLLM(),
        store=store,
        history_limit=2,
    )
    convo = service.start("c")
    for i in range(4):
        store.append(convo.id, Turn(f"q{i}", f"a{i}"))
    service.reply(convo.id, "latest")
    history, _ = rewriter.calls[0]
    assert history == [Turn("q2", "a2"), Turn("q3", "a3")]
