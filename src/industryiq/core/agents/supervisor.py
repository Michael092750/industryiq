"""Option C supervisor: enqueue waves, detect completion, synthesize.

The supervisor owns the run *policy*, not the work. It enqueues the plan's ready
nodes onto the shared :class:`TaskQueue`, then polls the blackboard until every
node has posted a result (or a run-level timeout trips), enqueuing each next
dependency wave as its inputs land. Workers -- possibly in other processes -- do
the claiming, running, reclaiming, and dead-lettering.

Completion detection is explicit because a competing-consumers queue has no native
"join": the supervisor knows the expected node-id set and watches the blackboard
namespace fill up. Re-running a completed plan is a no-op (the workers skip nodes
whose results already exist), so a run is resumable.

The ``on_poll`` seam replaces the sleep between polls: production leaves it unset
(the supervisor just sleeps while out-of-process workers churn); tests pass a
callback that drives in-process workers, making the whole distributed run
deterministic with no threads or timing.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable, Mapping
from typing import Any

from industryiq.core.agents.models import CapabilityResult, Plan, PlanNode, RunResult
from industryiq.core.agents.ports import Blackboard, PlanExecutor, RunLedger, TaskQueue
from industryiq.core.agents.synthesis import Synthesizer, collect_results
from industryiq.core.agents.worker import DEFAULT_QUEUE


class Supervisor(PlanExecutor):
    """Drive one distributed run: plan in, enqueue + await + synthesize, result out."""

    def __init__(
        self,
        queue: TaskQueue,
        blackboard: Blackboard,
        synthesizer: Synthesizer,
        *,
        ledger: RunLedger | None = None,
        queue_name: str = DEFAULT_QUEUE,
        run_timeout_s: float = 30.0,
        poll_interval_s: float = 0.1,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._queue = queue
        self._blackboard = blackboard
        self._synthesizer = synthesizer
        self._ledger = ledger
        self._queue_name = queue_name
        self._run_timeout_s = run_timeout_s
        self._poll_interval_s = poll_interval_s
        self._clock = clock
        self._sleep = sleep

    def execute(
        self, plan: Plan, *, on_poll: Callable[[], None] | None = None
    ) -> Mapping[str, CapabilityResult]:
        """Enqueue the plan wave by wave and await completion; return the results.

        Workers (possibly other processes) claim/run/reclaim/ack; this loop just
        enqueues newly-ready nodes and watches the blackboard fill up until every
        node has a result or ``run_timeout_s`` trips. ``on_poll`` replaces the
        inter-poll sleep (tests drive in-process workers through it).
        """
        all_ids = plan.node_ids()
        enqueued: set[str] = set()
        # Record the whole plan in the ledger (not just a count) so the run inspector
        # can list nodes + dependencies without a separate durable plan store.
        self._emit(
            plan.run_id,
            "plan_created",
            question=plan.question,
            nodes=[
                {"node_id": n.node_id, "capability": n.capability, "depends_on": list(n.depends_on)}
                for n in plan.nodes
            ],
        )
        self._enqueue_ready(plan, set(), enqueued)
        deadline = self._clock() + self._run_timeout_s
        while True:
            done = self._completed(plan)
            if done >= all_ids or self._clock() >= deadline:
                break
            self._enqueue_ready(plan, done, enqueued)
            if on_poll is not None:
                on_poll()
            else:
                self._sleep(self._poll_interval_s)
        return collect_results(plan, self._blackboard)

    def run(self, plan: Plan, *, on_poll: Callable[[], None] | None = None) -> RunResult:
        """``execute`` then synthesize the full :class:`RunResult`."""
        result = self._synthesizer.synthesize(plan, self.execute(plan, on_poll=on_poll))
        self._emit(
            plan.run_id,
            "run_completed",
            completed=len(result.completed),
            failed=len(result.failed),
        )
        return result

    def _enqueue_ready(self, plan: Plan, done: set[str], enqueued: set[str]) -> None:
        for node in plan.ready(done):
            if node.node_id in enqueued:
                continue
            self._queue.enqueue(
                self._queue_name,
                {
                    "run_id": plan.run_id,
                    "node_id": node.node_id,
                    "capability": node.capability,
                    "inputs": node.inputs,
                    "idempotency_key": _idempotency_key(plan.run_id, node),
                },
            )
            enqueued.add(node.node_id)
            self._emit(plan.run_id, "task_enqueued", node_id=node.node_id)

    def _completed(self, plan: Plan) -> set[str]:
        keys = self._blackboard.entries(plan.run_id).keys()
        return {node.node_id for node in plan.nodes if node.node_id in keys}

    def _emit(self, run_id: str, event: str, **fields: Any) -> None:
        if self._ledger is not None:
            self._ledger.append(run_id, {"event": event, **fields})


def _idempotency_key(run_id: str, node: PlanNode) -> str:
    """A deterministic key for a (run, node): identical logical work -> identical key."""
    raw = json.dumps(
        {
            "run_id": run_id,
            "node_id": node.node_id,
            "capability": node.capability,
            "inputs": node.inputs,
        },
        sort_keys=True,
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
