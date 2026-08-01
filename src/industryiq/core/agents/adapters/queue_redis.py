"""Redis Streams task queue: durable, cross-process, competing consumers.

One stream per queue (``taskq:{queue}``) with a single consumer group (default
``workers``). The supervisor ``XADD``s tasks; each worker ``XREADGROUP``s with its
own consumer name, so every task goes to exactly one worker and stays in the
group's pending list (PEL) until acked -- at-least-once delivery. :meth:`ack` both
``XACK``s (clears the pending entry) and ``XDEL``s (drops it from the stream), so
an empty stream means all work is done and ``XLEN`` is the outstanding count.

Recovery: a worker that dies leaves its task in the PEL forever, because
``XREADGROUP ">"`` only ever delivers *new* entries. :meth:`reclaim` closes that
gap -- it re-delivers (``XCLAIM``) entries that have sat idle past a threshold to a
live worker, incrementing their delivery count; :meth:`dead_letter` moves a task
that keeps failing to a ``taskq:{queue}:dead`` stream. :meth:`inflight` exposes the
PEL (``XPENDING``) so a slow worker can be told from a dead one.

The group is created lazily from id ``0`` (``MKSTREAM``) so tasks enqueued before
any worker started are still delivered; re-creation is a harmless ``BUSYGROUP``.

``redis`` is a ``redis.Redis`` client (``decode_responses=True``), typed ``Any``
for the same reason as the other Redis adapters (redis-py stubs don't model
``decode_responses``; see the mypy overrides in pyproject).
"""

import json
from typing import Any

from redis.exceptions import ResponseError

from industryiq.core.agents.models import InFlight, Task
from industryiq.core.agents.ports import TaskQueue

_KEY_PREFIX = "taskq:"


class RedisTaskQueue(TaskQueue):
    """Competing-consumers task queue backed by a Redis stream + consumer group."""

    def __init__(
        self, redis: Any, *, group: str = "workers", ttl_seconds: int | None = None
    ) -> None:
        self._redis = redis
        self._group = group
        self._ttl = ttl_seconds

    def _key(self, queue: str) -> str:
        return f"{_KEY_PREFIX}{queue}"

    def _ensure_group(self, key: str) -> None:
        # id="0" so the group treats pre-existing entries as undelivered; MKSTREAM
        # creates the stream if a worker starts before the first enqueue.
        try:
            self._redis.xgroup_create(key, self._group, id="0", mkstream=True)
        except ResponseError as exc:
            if "BUSYGROUP" not in str(exc):
                raise

    def _pending(self, key: str) -> dict[str, tuple[str, float, int]]:
        """Return the PEL as ``{id: (consumer, idle_ms, times_delivered)}``."""
        self._ensure_group(key)
        entries = self._redis.xpending_range(key, self._group, min="-", max="+", count=1000)
        return {
            entry["message_id"]: (
                entry["consumer"],
                float(entry["time_since_delivered"]),
                int(entry["times_delivered"]),
            )
            for entry in entries
        }

    def enqueue(self, queue: str, payload: dict[str, Any]) -> Task:
        key = self._key(queue)
        self._ensure_group(key)
        task_id: str = self._redis.xadd(key, {"payload": json.dumps(payload)})
        if self._ttl is not None:
            self._redis.expire(key, self._ttl)
        return Task(id=task_id, payload=payload)

    def claim(self, queue: str, consumer: str, *, count: int = 1) -> list[Task]:
        key = self._key(queue)
        self._ensure_group(key)
        resp = self._redis.xreadgroup(self._group, consumer, {key: ">"}, count=count)
        if not resp:
            return []
        _stream_key, entries = resp[0]
        return [
            Task(id=entry_id, payload=json.loads(fields["payload"])) for entry_id, fields in entries
        ]

    def ack(self, queue: str, task: Task) -> None:
        key = self._key(queue)
        self._redis.xack(key, self._group, task.id)
        self._redis.xdel(key, task.id)

    def pending(self, queue: str) -> int:
        return int(self._redis.xlen(self._key(queue)))

    def inflight(self, queue: str) -> list[InFlight]:
        key = self._key(queue)
        out: list[InFlight] = []
        for msg_id, (consumer, idle_ms, delivered) in self._pending(key).items():
            entries = self._redis.xrange(key, min=msg_id, max=msg_id, count=1)
            if not entries:  # acked+deleted between XPENDING and XRANGE
                continue
            _entry_id, fields = entries[0]
            out.append(
                InFlight(
                    task=Task(id=msg_id, payload=json.loads(fields["payload"]), attempt=delivered),
                    consumer=consumer,
                    idle_ms=idle_ms,
                    delivery_count=delivered,
                )
            )
        return out

    def reclaim(
        self, queue: str, consumer: str, *, min_idle_ms: float, count: int = 10
    ) -> list[Task]:
        key = self._key(queue)
        pending = self._pending(key)
        idle = [msg_id for msg_id, (_c, ms, _d) in pending.items() if ms >= min_idle_ms]
        if not idle:
            return []
        # XCLAIM reassigns these entries to ``consumer`` and (without JUSTID)
        # increments each one's delivery counter by one.
        claimed = self._redis.xclaim(key, self._group, consumer, int(min_idle_ms), idle[:count])
        tasks: list[Task] = []
        for msg_id, fields in claimed:
            if not fields:  # entry was deleted; skip
                continue
            prior = pending.get(msg_id)
            attempt = (prior[2] + 1) if prior is not None else 1
            tasks.append(Task(id=msg_id, payload=json.loads(fields["payload"]), attempt=attempt))
        return tasks

    def dead_letter(self, queue: str, task: Task, *, reason: str) -> None:
        key = self._key(queue)
        dead_key = f"{key}:dead"
        self._redis.xadd(
            dead_key,
            {"payload": json.dumps(task.payload), "reason": reason, "attempt": str(task.attempt)},
        )
        if self._ttl is not None:
            self._redis.expire(dead_key, self._ttl)
        self._redis.xack(key, self._group, task.id)
        self._redis.xdel(key, task.id)

    def dead(self, queue: str) -> list[Task]:
        dead_key = f"{self._key(queue)}:dead"
        return [
            Task(id=entry_id, payload=json.loads(fields["payload"]), attempt=int(fields["attempt"]))
            for entry_id, fields in self._redis.xrange(dead_key)
        ]
