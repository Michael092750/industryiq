"""In-memory run ledger: the default (no Redis) and the test double.

A dict of ``run_id -> [event, ...]``, appended under a lock so worker threads in
one process can log concurrently. Process-local and lost on restart -- which is why
the Redis-backed ledger exists for a real multi-process run.
"""

import threading
from typing import Any

from industryiq.core.agents.ports import RunLedger


class InMemoryRunLedger(RunLedger):
    """A list-per-run event log for tests and single-process local development."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._events: dict[str, list[dict[str, Any]]] = {}

    def append(self, run_id: str, event: dict[str, Any]) -> None:
        with self._lock:
            self._events.setdefault(run_id, []).append(dict(event))

    def events(self, run_id: str) -> list[dict[str, Any]]:
        with self._lock:
            return [dict(event) for event in self._events.get(run_id, ())]
