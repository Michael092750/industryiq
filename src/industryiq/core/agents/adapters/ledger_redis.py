"""Redis Streams run ledger: append-only, cross-process, TTL'd.

One stream per run (``runlog:{run_id}``); each event is one ``XADD`` of a
JSON-encoded dict, read back in append order with ``XRANGE``. A stream (not a list)
so the ordering and entry ids are native, and an optional TTL, refreshed on each
append, self-evicts a finished run's timeline -- mirroring the blackboard's per-run
TTL.

``redis`` is a ``redis.Redis`` client (``decode_responses=True``), typed ``Any``
because redis-py's stubs don't model ``decode_responses`` (see the mypy overrides).
"""

import json
from typing import Any

from industryiq.core.agents.ports import RunLedger

_KEY_PREFIX = "runlog:"


class RedisRunLedger(RunLedger):
    """Append-only per-run event log backed by one Redis stream per run."""

    def __init__(self, redis: Any, *, ttl_seconds: int | None = None) -> None:
        self._redis = redis
        self._ttl = ttl_seconds

    def _key(self, run_id: str) -> str:
        return f"{_KEY_PREFIX}{run_id}"

    def append(self, run_id: str, event: dict[str, Any]) -> None:
        key = self._key(run_id)
        self._redis.xadd(key, {"event": json.dumps(event)})
        if self._ttl is not None:
            self._redis.expire(key, self._ttl)

    def events(self, run_id: str) -> list[dict[str, Any]]:
        entries = self._redis.xrange(self._key(run_id))
        return [json.loads(fields["event"]) for _entry_id, fields in entries]
