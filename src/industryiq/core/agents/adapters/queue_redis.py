"""Redis Streams task queue: durable, cross-process, competing consumers.

One stream per queue (``taskq:{queue}``) with a single consumer group (default
``workers``). The supervisor ``XADD``s tasks; each worker ``XREADGROUP``s with its
own consumer name, so every task goes to exactly one worker and stays in the
group's pending list until acked -- at-least-once delivery. :meth:`ack` both
``XACK``s (clears the pending entry) and ``XDEL``s (drops it from the stream), so
an empty stream means all work is done and ``XLEN`` is the outstanding count.

The group is created lazily from id ``0`` (``MKSTREAM``) so tasks enqueued before
any worker started are still delivered; re-creation is a harmless ``BUSYGROUP``.

``redis`` is a ``redis.Redis`` client (``decode_responses=True``), typed ``Any``
for the same reason as the other Redis adapters (redis-py stubs don't model
``decode_responses``; see the mypy overrides in pyproject).
"""

import json
from typing import Any

from redis.exceptions import ResponseError

from industryiq.core.agents.models import Task
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
