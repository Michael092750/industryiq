"""Option C worker: claim tasks, run capabilities, post results -- with recovery.

A worker pulls tasks off the shared :class:`TaskQueue`, runs the named capability,
and writes the result to the blackboard keyed by node id. Two behaviours make the
distributed run resilient:

* **Reclaim** -- when there is no new work, the worker reclaims tasks a dead peer
  left unacked past the idle threshold, so a crashed worker's task is finished by a
  live one rather than lost.
* **Idempotent skip** -- if a node's result is already on the blackboard, the task
  is acked without re-running it (memoized resume). Combined with keyed writes, a
  reclaimed/duplicated task is free of double effects.

A capability that raises (or the demo :data:`FailureHook`) leaves the task *unacked*
-- it stays outstanding and becomes reclaimable, which is exactly at-least-once
recovery. A task reclaimed more than ``max_attempts`` times is dead-lettered instead
of looping forever.
"""

from __future__ import annotations

import threading
from collections.abc import Mapping
from typing import Any

from industryiq.core.agents.capabilities import FailureHook, no_failure
from industryiq.core.agents.models import Task
from industryiq.core.agents.ports import Blackboard, Capability, RunLedger, TaskQueue

DEFAULT_QUEUE = "agents"


class Worker:
    """One competing consumer of the agents queue."""

    def __init__(
        self,
        queue: TaskQueue,
        registry: Mapping[str, Capability],
        blackboard: Blackboard,
        *,
        consumer: str,
        ledger: RunLedger | None = None,
        failure_hook: FailureHook = no_failure,
        queue_name: str = DEFAULT_QUEUE,
        max_attempts: int = 3,
        reclaim_min_idle_ms: float = 5000.0,
        batch: int = 4,
    ) -> None:
        self._queue = queue
        self._registry = registry
        self._blackboard = blackboard
        self._consumer = consumer
        self._ledger = ledger
        self._failure_hook = failure_hook
        self._queue_name = queue_name
        self._max_attempts = max_attempts
        self._reclaim_min_idle_ms = reclaim_min_idle_ms
        self._batch = batch

    def run_once(self) -> int:
        """Process one batch: claim new work, or reclaim abandoned work. Returns the count."""
        tasks = self._queue.claim(self._queue_name, self._consumer, count=self._batch)
        if not tasks:
            tasks = self._queue.reclaim(
                self._queue_name,
                self._consumer,
                min_idle_ms=self._reclaim_min_idle_ms,
                count=self._batch,
            )
        for task in tasks:
            self._process(task)
        return len(tasks)

    def run_forever(self, stop: threading.Event, *, idle_sleep: float = 0.2) -> None:
        """Loop :meth:`run_once` until ``stop`` is set, backing off when idle."""
        while not stop.is_set():
            if self.run_once() == 0:
                stop.wait(idle_sleep)

    def _process(self, task: Task) -> None:
        run_id = str(task.payload["run_id"])
        node_id = str(task.payload["node_id"])
        if task.attempt > self._max_attempts:
            self._queue.dead_letter(
                self._queue_name, task, reason=f"exceeded {self._max_attempts} attempts"
            )
            self._emit(run_id, "task_dead_lettered", node_id=node_id, attempt=task.attempt)
            return
        if self._blackboard.read(run_id, node_id) is not None:
            self._queue.ack(self._queue_name, task)  # memoized: already done, don't re-run
            self._emit(run_id, "task_skipped", node_id=node_id)
            return
        try:
            self._failure_hook(node_id)
            result = self._registry[str(task.payload["capability"])].run(task.payload["inputs"])
        except Exception as exc:
            # A failed attempt must NOT ack -- the task stays outstanding and another
            # worker will reclaim it. That is the crash-recovery path.
            self._emit(
                run_id, "task_failed", node_id=node_id, consumer=self._consumer, error=str(exc)
            )
            return
        self._blackboard.write(run_id, node_id, result.as_dict())
        self._emit(run_id, "result_written", node_id=node_id, consumer=self._consumer)
        self._queue.ack(self._queue_name, task)

    def _emit(self, run_id: str, event: str, **fields: Any) -> None:
        if self._ledger is not None:
            self._ledger.append(run_id, {"event": event, "worker": self._consumer, **fields})
