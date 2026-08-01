"""In-memory task queue: the default (no Redis) and the test double.

Mirrors the competing-consumers semantics of the Redis Streams queue without a
broker: each queue keeps an ordered list of undelivered tasks and a map of
in-flight (claimed-but-unacked) claims. :meth:`claim` moves tasks from one to the
other; :meth:`ack` drops them; :meth:`reclaim` re-delivers a claim that has been
idle too long (the crash-recovery a real broker gets from ``XAUTOCLAIM``). A lock
makes the transitions safe when workers run as threads in one process -- the only
concurrency this double needs (the Redis adapter gets atomicity from Redis).

A clock is injected (``time.monotonic`` by default) so a test can drive idle time
deterministically -- reclaim-after-idle is a *timing* behaviour, and a fake clock
turns it into a reproducible unit test instead of a ``sleep``.
"""

import threading
import time
import uuid
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any

from industryiq.core.agents.models import InFlight, Task
from industryiq.core.agents.ports import TaskQueue


@dataclass
class _Claim:
    """A task currently held by a consumer, with when it was last delivered."""

    task: Task
    consumer: str
    claimed_at: float
    delivery_count: int


class InMemoryTaskQueue(TaskQueue):
    """A deque-backed competing-consumers queue for tests and local development."""

    def __init__(self, *, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._lock = threading.Lock()
        self._undelivered: dict[str, deque[Task]] = {}
        self._in_flight: dict[str, dict[str, _Claim]] = {}
        self._dead: dict[str, list[Task]] = {}

    def enqueue(self, queue: str, payload: dict[str, Any]) -> Task:
        task = Task(id=uuid.uuid4().hex, payload=payload)
        with self._lock:
            self._undelivered.setdefault(queue, deque()).append(task)
        return task

    def claim(self, queue: str, consumer: str, *, count: int = 1) -> list[Task]:
        claimed: list[Task] = []
        with self._lock:
            waiting = self._undelivered.get(queue)
            in_flight = self._in_flight.setdefault(queue, {})
            now = self._clock()
            while waiting and len(claimed) < count:
                task = waiting.popleft()
                in_flight[task.id] = _Claim(task, consumer, claimed_at=now, delivery_count=1)
                claimed.append(task)
        return claimed

    def ack(self, queue: str, task: Task) -> None:
        with self._lock:
            self._in_flight.get(queue, {}).pop(task.id, None)

    def pending(self, queue: str) -> int:
        with self._lock:
            return len(self._undelivered.get(queue, ())) + len(self._in_flight.get(queue, {}))

    def inflight(self, queue: str) -> list[InFlight]:
        with self._lock:
            now = self._clock()
            return [
                InFlight(
                    task=replace(claim.task, attempt=claim.delivery_count),
                    consumer=claim.consumer,
                    idle_ms=(now - claim.claimed_at) * 1000.0,
                    delivery_count=claim.delivery_count,
                )
                for claim in self._in_flight.get(queue, {}).values()
            ]

    def reclaim(
        self, queue: str, consumer: str, *, min_idle_ms: float, count: int = 10
    ) -> list[Task]:
        reclaimed: list[Task] = []
        with self._lock:
            now = self._clock()
            for claim in self._in_flight.get(queue, {}).values():
                if len(reclaimed) >= count:
                    break
                if (now - claim.claimed_at) * 1000.0 < min_idle_ms:
                    continue
                # Re-deliver: hand it to the reclaiming consumer, reset its idle
                # clock, and bump the delivery count so a poison task can be
                # dead-lettered after enough reclaims.
                claim.consumer = consumer
                claim.claimed_at = now
                claim.delivery_count += 1
                claim.task = replace(claim.task, attempt=claim.delivery_count)
                reclaimed.append(claim.task)
        return reclaimed

    def dead_letter(self, queue: str, task: Task, *, reason: str) -> None:
        with self._lock:
            self._in_flight.get(queue, {}).pop(task.id, None)
            self._dead.setdefault(queue, []).append(task)

    def dead(self, queue: str) -> list[Task]:
        with self._lock:
            return list(self._dead.get(queue, ()))
