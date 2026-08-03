"""Tests for AgentTurnOrchestrator -- the chat->agents bridge for complex turns.

Drives the adapter with a fake planner + a real in-process LocalExecutor over a
stub capability, so the whole rewrite -> plan -> execute -> stream path runs offline.
"""

from industryiq.core.agents.adapters.blackboard_memory import InMemoryBlackboard
from industryiq.core.agents.executor_local import LocalExecutor
from industryiq.core.agents.models import CapabilityResult, Plan, PlanNode
from industryiq.core.agents.synthesis import Synthesizer
from industryiq.core.chat.adapters.orchestration import AgentTurnOrchestrator
from industryiq.core.chat.models import StreamEnd, StreamStart, StreamStatus, StreamToken
from industryiq.core.generation import FakeLLM
from industryiq.core.retrieval.adapters.rewriting import NoOpQueryRewriter


class _StubCapability:
    name = "retrieve"
    description = "corpus lookup"

    def run(self, inputs: dict[str, object]) -> CapabilityResult:
        industry = inputs.get("industry", "?")
        return CapabilityResult(
            summary=f"about {industry}", sources=[{"source": "doc-X", "score": 1.0}]
        )


class _FakePlanner:
    def __init__(self, nodes: tuple[PlanNode, ...]) -> None:
        self._nodes = nodes
        self.calls: list[tuple[str, str, dict[str, str]]] = []

    def plan(self, run_id: str, question: str, capabilities: dict[str, str]) -> Plan:
        self.calls.append((run_id, question, dict(capabilities)))
        return Plan(run_id=run_id, question=question, nodes=self._nodes)


def _adapter(planner: _FakePlanner) -> AgentTurnOrchestrator:
    registry = {"retrieve": _StubCapability()}
    executor = LocalExecutor(registry, InMemoryBlackboard(), Synthesizer())
    return AgentTurnOrchestrator(
        rewriter=NoOpQueryRewriter(),
        planner=planner,
        registry=registry,
        executor=executor,
        synthesizer=Synthesizer(FakeLLM("final answer")),
    )


def test_orchestrator_plans_executes_and_streams() -> None:
    planner = _FakePlanner(
        (
            PlanNode("n1", "retrieve", {"industry": "AI"}),
            PlanNode("n2", "retrieve", {"industry": "finance"}),
        )
    )
    events = list(_adapter(planner).run_stream([], "compare AI and finance"))

    # planner saw the (rewritten) standalone question + the capability catalog
    assert planner.calls[0][1] == "compare AI and finance"
    assert "retrieve" in planner.calls[0][2]
    # a planning status, a StreamStart carrying the merged/de-duped sources, then tokens
    assert any(isinstance(e, StreamStatus) and e.phase == "planning" for e in events)
    start = next(e for e in events if isinstance(e, StreamStart))
    assert [hit.metadata["source"] for hit in start.hits] == ["doc-X"]
    tokens = "".join(e.text for e in events if isinstance(e, StreamToken))
    assert tokens == "final answer"


def test_orchestrator_does_not_emit_stream_end() -> None:
    planner = _FakePlanner((PlanNode("n1", "retrieve", {}),))
    events = list(_adapter(planner).run_stream([], "q"))
    assert not any(isinstance(e, StreamEnd) for e in events)  # ChatService owns StreamEnd
