"""Tests for the industry-analysis capability and the demo failure hook."""

import pytest

from industryiq.core.agents.capabilities import (
    CrashOnceHook,
    IndustryAnalysisCapability,
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
