"""Domain types for the multi-agent coordination substrate.

Small, immutable value objects shared across the agents module -- no behavior, no
I/O -- so ports, adapters, and (later) the agents themselves can depend on them
without coupling to a backend.
"""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Task:
    """One unit of work on a :class:`~industryiq.core.agents.ports.TaskQueue`.

    ``id`` is assigned by the queue on enqueue (a Redis stream entry id, or a uuid
    in the in-memory queue) and is what :meth:`TaskQueue.ack` references. ``payload``
    is an arbitrary JSON-serializable dict -- the queue treats it as opaque; the
    supervisor and workers agree on its shape.
    """

    id: str
    payload: dict[str, Any]
