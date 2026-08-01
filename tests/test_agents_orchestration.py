"""Executor tests: Option B (local) vs Option C (supervisor + workers).

The headline is ``test_supervisor_resumes_after_a_worker_crash`` -- the demo, as a
deterministic unit test: a worker dies mid-node, a peer reclaims the task, and the
run still completes. The in-memory queue's injected clock lets us cross the reclaim
idle threshold with no sleeps.
"""

import pytest

from industryiq.core.agents.adapters.blackboard_memory import InMemoryBlackboard
from industryiq.core.agents.adapters.queue_memory import InMemoryTaskQueue
from industryiq.core.agents.capabilities import CrashOnceHook, WorkerCrash
from industryiq.core.agents.executor_local import LocalExecutor
from industryiq.core.agents.models import CapabilityResult, Plan, PlanNode
from industryiq.core.agents.supervisor import Supervisor
from industryiq.core.agents.synthesis import Synthesizer
from industryiq.core.agents.worker import DEFAULT_QUEUE, Worker


class _StubCapability:
    """A deterministic capability: echoes its industry, cites one source."""

    name = "industry_analysis"
    description = "stub"

    def run(self, inputs: dict[str, object]) -> CapabilityResult:
        industry = inputs.get("industry", "?")
        return CapabilityResult(
            summary=f"analysis of {industry}", sources=[{"source": f"doc-{industry}", "score": 1.0}]
        )


class _CountingCapability:
    name = "industry_analysis"
    description = "counts runs"

    def __init__(self) -> None:
        self.calls = 0

    def run(self, inputs: dict[str, object]) -> CapabilityResult:
        self.calls += 1
        return CapabilityResult(summary="done")


class _AlwaysCrash:
    def __call__(self, node_id: str) -> None:
        raise WorkerCrash(f"always crash {node_id}")


class _Clock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += seconds


def _fanout_plan() -> Plan:
    return Plan(
        run_id="run1",
        question="compare AI and finance",
        nodes=(
            PlanNode("n1", "industry_analysis", {"industry": "AI"}),
            PlanNode("n2", "industry_analysis", {"industry": "finance"}),
        ),
    )


# --- Option B: LocalExecutor ------------------------------------------------------


def test_local_executor_completes_a_dependency_chain() -> None:
    blackboard = InMemoryBlackboard()
    registry = {"industry_analysis": _StubCapability()}
    plan = Plan(
        "r",
        "q",
        (
            PlanNode("n1", "industry_analysis", {"industry": "AI"}),
            PlanNode("n2", "industry_analysis", {"industry": "finance"}, depends_on=("n1",)),
        ),
    )
    result = LocalExecutor(registry, blackboard, Synthesizer()).run(plan)
    assert set(result.completed) == {"n1", "n2"}
    assert result.failed == ()
    assert blackboard.read("r", "n2") is not None


def test_local_executor_loses_the_whole_run_on_a_crash() -> None:
    blackboard = InMemoryBlackboard()
    registry = {"industry_analysis": _StubCapability()}
    plan = Plan("r", "q", (PlanNode("n1", "industry_analysis", {}),))
    with pytest.raises(WorkerCrash):
        LocalExecutor(registry, blackboard, Synthesizer(), failure_hook=CrashOnceHook()).run(plan)
    assert blackboard.read("r", "n1") is None  # nothing persisted -> run lost, no recovery


# --- Option C: Supervisor + Worker ------------------------------------------------


def test_supervisor_completes_a_fan_out() -> None:
    queue = InMemoryTaskQueue()
    blackboard = InMemoryBlackboard()
    registry = {"industry_analysis": _StubCapability()}
    worker = Worker(queue, registry, blackboard, consumer="w1")
    supervisor = Supervisor(queue, blackboard, Synthesizer(), run_timeout_s=1e9)
    result = supervisor.run(_fanout_plan(), on_poll=worker.run_once)
    assert set(result.completed) == {"n1", "n2"}
    assert "doc-AI" in {s["source"] for s in result.sources}


def test_supervisor_resumes_after_a_worker_crash() -> None:
    clock = _Clock()
    queue = InMemoryTaskQueue(clock=clock)
    blackboard = InMemoryBlackboard()
    registry = {"industry_analysis": _StubCapability()}
    # One shared hook so node n1 crashes exactly once (its first attempt, on w1).
    hook = CrashOnceHook(nodes={"n1"})

    def make(name: str) -> Worker:
        return Worker(
            queue, registry, blackboard, consumer=name, failure_hook=hook, reclaim_min_idle_ms=5000
        )

    w1, w2 = make("w1"), make("w2")
    supervisor = Supervisor(queue, blackboard, Synthesizer(), clock=clock, run_timeout_s=1e9)

    def on_poll() -> None:
        w1.run_once()  # claims n1 (crashes, left unacked) and n2 (completes)
        clock.advance(6.0)  # n1 now idle past the 5s reclaim threshold
        w2.run_once()  # reclaims n1 and finishes it

    result = supervisor.run(_fanout_plan(), on_poll=on_poll)
    assert set(result.completed) == {"n1", "n2"}  # crash survived -> run completed
    assert result.failed == ()
    assert queue.pending(DEFAULT_QUEUE) == 0  # everything acked


def test_worker_skips_an_already_completed_node() -> None:
    queue = InMemoryTaskQueue()
    blackboard = InMemoryBlackboard()
    capability = _CountingCapability()
    blackboard.write("r", "n1", CapabilityResult(summary="precomputed").as_dict())
    queue.enqueue(
        DEFAULT_QUEUE,
        {"run_id": "r", "node_id": "n1", "capability": "industry_analysis", "inputs": {}},
    )
    Worker(queue, {"industry_analysis": capability}, blackboard, consumer="w1").run_once()
    assert capability.calls == 0  # memoized: not re-run
    assert queue.pending(DEFAULT_QUEUE) == 0  # but acked (task consumed)


def test_worker_dead_letters_a_poison_task() -> None:
    clock = _Clock()
    queue = InMemoryTaskQueue(clock=clock)
    blackboard = InMemoryBlackboard()
    worker = Worker(
        queue,
        {"industry_analysis": _StubCapability()},
        blackboard,
        consumer="w1",
        failure_hook=_AlwaysCrash(),
        max_attempts=2,
        reclaim_min_idle_ms=1000,
    )
    queue.enqueue(
        DEFAULT_QUEUE,
        {"run_id": "r", "node_id": "n1", "capability": "industry_analysis", "inputs": {}},
    )
    worker.run_once()  # attempt 1: crash, unacked
    clock.advance(2.0)
    worker.run_once()  # attempt 2: reclaim, crash, unacked
    clock.advance(2.0)
    worker.run_once()  # attempt 3 > max_attempts -> dead-letter
    assert len(queue.dead(DEFAULT_QUEUE)) == 1
    assert blackboard.read("r", "n1") is None
