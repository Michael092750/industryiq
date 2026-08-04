"""Tests for the capabilities (industry-analysis, web-search) and the failure hook."""

from types import SimpleNamespace

import pytest

from industryiq.core.agents.capabilities import (
    CrashOnceHook,
    IndustryAnalysisCapability,
    WebSearchCapability,
    WorkerCrash,
)
from industryiq.core.agents.ports import Capability
from industryiq.core.generation import FakeLLM
from industryiq.core.vectorstore import Hit


class _FakeCorpus:
    """Records the query it was asked, returns a fixed hit list (CorpusRetriever shape)."""

    def __init__(self, hits: list[Hit]) -> None:
        self._hits = hits
        self.last_query: str | None = None

    def retrieve_corpus(self, question: str, k: int = 6) -> list[Hit]:
        self.last_query = question
        return self._hits


def test_capability_returns_grounded_cited_envelope() -> None:
    hits = [
        Hit(id="1", score=0.91, metadata={"text": "AI market is large", "source": "McKinsey"}),
        Hit(id="2", score=0.80, metadata={"text": "growth continues", "title": "BCG report"}),
    ]
    cap = IndustryAnalysisCapability(_FakeCorpus(hits), FakeLLM("grounded answer"))
    result = cap.run({"industry": "AI", "question": "how big is the market?"})
    assert result.summary == "grounded answer"
    assert result.data == {"industry": "AI", "chunks": 2}
    assert [s["source"] for s in result.sources] == ["McKinsey", "BCG report"]


def test_capability_scopes_query_by_industry() -> None:
    corpus = _FakeCorpus([])
    cap = IndustryAnalysisCapability(corpus, FakeLLM())
    cap.run({"industry": "healthcare", "question": "adoption rate?"})
    assert corpus.last_query == "healthcare: adoption rate?"


def test_capability_satisfies_the_port() -> None:
    assert isinstance(IndustryAnalysisCapability(_FakeCorpus([]), FakeLLM()), Capability)


class _FakeMessages:
    """Records create() calls, returns a canned message (Anthropic client shape)."""

    def __init__(self, message: object) -> None:
        self._message = message
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        return self._message


class _FakeAnthropic:
    def __init__(self, message: object) -> None:
        self.messages = _FakeMessages(message)


def _web_message(content: list[object], stop_reason: str = "end_turn") -> SimpleNamespace:
    return SimpleNamespace(content=content, stop_reason=stop_reason)


def test_web_search_capability_extracts_answer_and_deduped_sources() -> None:
    search_result = SimpleNamespace(
        type="web_search_tool_result",
        content=[
            SimpleNamespace(type="web_search_result", url="https://a.com", title="A"),
            SimpleNamespace(type="web_search_result", url="https://b.com", title="B"),
        ],
    )
    answer = SimpleNamespace(
        type="text",
        text="the web answer",
        citations=[SimpleNamespace(url="https://a.com", title="A")],  # dup of a result
    )
    client = _FakeAnthropic(_web_message([search_result, answer]))
    result = WebSearchCapability(model_id="claude-sonnet-4-6", client=client).run(
        {"question": "latest AI chip news?"}
    )
    assert result.summary == "the web answer"
    assert [s["source"] for s in result.sources] == ["https://a.com", "https://b.com"]  # deduped


def test_web_search_capability_declares_the_server_tool() -> None:
    client = _FakeAnthropic(_web_message([]))
    WebSearchCapability(model_id="claude-sonnet-4-6", client=client, max_searches=3).run(
        {"question": "q"}
    )
    call = client.messages.calls[0]
    tool = call["tools"][0]  # type: ignore[index]
    assert tool["type"] == "web_search_20260209"
    assert tool["name"] == "web_search"
    assert tool["max_uses"] == 3
    assert call["messages"][0]["content"] == "q"  # type: ignore[index]


def test_web_search_capability_satisfies_the_port() -> None:
    capability = WebSearchCapability(model_id="m", client=_FakeAnthropic(_web_message([])))
    assert isinstance(capability, Capability)


def test_crash_once_hook_fires_once_per_node() -> None:
    hook = CrashOnceHook()
    with pytest.raises(WorkerCrash):
        hook("n1")
    hook("n1")  # second time on the same node: passes through
    with pytest.raises(WorkerCrash):
        hook("n2")  # a different node still crashes its first time


def test_no_failure_targeting_limits_crashes() -> None:
    hook = CrashOnceHook(nodes={"n2"})
    hook("n1")  # not targeted -> never crashes
    with pytest.raises(WorkerCrash):
        hook("n2")
