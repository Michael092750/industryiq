"""Option B executor: run the plan in-process, concurrently, with no recovery.

The single-process baseline. It schedules the plan in dependency waves and runs
each wave's nodes on a thread pool -- real parallelism, because the capabilities
are I/O-bound (they wait on the LLM / retrieval), so the GIL is not the ceiling.
Each result is written to the blackboard keyed by node id, then :class:`Synthesizer`
composes the answer.

What it deliberately lacks is the whole point of the demo: **no queue, no reclaim,
no durability**. A node that crashes propagates out of :meth:`run` and the entire
run is lost -- there is nothing to resume from. That is the "B breaks" beat; Option
C (:mod:`industryiq.core.agents.supervisor`) survives the same crash.
"""

from __future__ import annotations

from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor

from industryiq.core.agents.capabilities import FailureHook, no_failure
from industryiq.core.agents.grounding import check_node_result, node_question
from industryiq.core.agents.models import CapabilityResult, Plan, PlanNode, RunResult
from industryiq.core.agents.ports import Blackboard, Capability, PlanExecutor, RunLedger
from industryiq.core.agents.synthesis import Synthesizer, collect_results
from industryiq.core.grounding import GroundingGate


class LocalExecutor(PlanExecutor):
    """Run a :class:`Plan` in one process, one dependency wave at a time."""

    def __init__(
        self,
        registry: Mapping[str, Capability],
        blackboard: Blackboard,
        synthesizer: Synthesizer,
        *,
        ledger: RunLedger | None = None,
        grounding: GroundingGate | None = None,
        failure_hook: FailureHook = no_failure,
        max_workers: int = 8,
    ) -> None:
        self._registry = registry
        self._blackboard = blackboard
        self._synthesizer = synthesizer
        self._ledger = ledger
        # Checked between running a node and posting it, so an ungrounded subtask
        # never becomes synthesis input. ``None`` => no gate (previous behaviour).
        self._grounding = grounding
        self._failure_hook = failure_hook
        self._max_workers = max_workers

    def execute(self, plan: Plan) -> Mapping[str, CapabilityResult]:
        """Run the plan's waves in-process; return the per-node results.

        A node crash propagates out (via ``future.result()``) -- Option B has no
        recovery, so the whole run is lost. That is the "B breaks" beat.
        """
        done: set[str] = set()
        while True:
            wave = plan.ready(done)
            if not wave:
                break
            with ThreadPoolExecutor(max_workers=min(self._max_workers, len(wave))) as pool:
                futures = {pool.submit(self._run_node, plan, node): node for node in wave}
                for future in futures:
                    future.result()  # re-raises a node crash
                    done.add(futures[future].node_id)
        return collect_results(plan, self._blackboard)

    def run(self, plan: Plan) -> RunResult:
        """``execute`` then synthesize the full :class:`RunResult`."""
        return self._synthesizer.synthesize(plan, self.execute(plan))

    def _run_node(self, plan: Plan, node: PlanNode) -> None:
        self._failure_hook(node.node_id)  # demo crash -> propagates -> run aborts
        result = self._registry[node.capability].run(node.inputs)
        question = node_question(node.inputs, plan.question)
        result = check_node_result(self._grounding, question, result)
        self._blackboard.write(plan.run_id, node.node_id, result.as_dict())
        if self._ledger is not None:
            self._ledger.append(
                plan.run_id,
                {"event": "result_written", "node_id": node.node_id, "executor": "local"},
            )
