"""In-memory task queue: the default (no Redis) and the test double.

Mirrors the competing-consumers semantics of the Redis Streams queue without a
broker: each queue keeps an ordered list of undelivered tasks and a map of
in-flight (claimed-but-unacked) tasks. :meth:`claim` moves tasks from one to the
other; :meth:`ack` drops them. A lock makes claim/ack safe when workers run as
threads in one process -- the only concurrency this double needs (the Redis
adapter gets atomicity from Redis itself).

What it deliberately omits vs. a real broker: reclaiming tasks from a crashed
consumer (Streams' XAUTOCLAIM). In-process there is no crash to recover from.
"""

import threading
import uuid
from collections import deque
from typing import Any

from industryiq.core.agents.models import Task
from industryiq.core.agents.ports import TaskQueue


class InMemoryTaskQueue(TaskQueue):
    """A deque-backed competing-consumers queue for tests and local development."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._undelivered: dict[str, deque[Task]] = {}
        self._in_flight: dict[str, dict[str, Task]] = {}

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
            while waiting and len(claimed) < count:
                task = waiting.popleft()
                in_flight[task.id] = task
                claimed.append(task)
        return claimed

    def ack(self, queue: str, task: Task) -> None:
        with self._lock:
            self._in_flight.get(queue, {}).pop(task.id, None)

    def pending(self, queue: str) -> int:
        with self._lock:
            return len(self._undelivered.get(queue, ())) + len(self._in_flight.get(queue, {}))
