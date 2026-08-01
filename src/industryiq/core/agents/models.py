"""Domain types for the multi-agent coordination substrate.

Small, immutable value objects shared across the agents module -- no I/O, and no
behavior beyond pure helpers -- so ports, adapters, and the agents themselves can
depend on them without coupling to a backend. Everything here is
JSON-serializable, because these values travel through the task queue (payloads),
the blackboard (results), and the run ledger (events).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Task:
    """One unit of work on a :class:`~industryiq.core.agents.ports.TaskQueue`.

    ``id`` is assigned by the queue on enqueue (a Redis stream entry id, or a uuid
    in the in-memory queue) and is what :meth:`TaskQueue.ack` references. ``payload``
    is an arbitrary JSON-serializable dict -- the queue treats it as opaque; the
    supervisor and workers agree on its shape.

    ``attempt`` is how many times this task has been *delivered*: 1 on a fresh
    claim, and incremented when a reclaim picks it up after a previous consumer
    left it unacked (a crash). A worker reads it to dead-letter a poison task
    instead of looping on it forever.
    """

    id: str
    payload: dict[str, Any]
    attempt: int = 1


@dataclass(frozen=True)
class InFlight:
    """A claimed-but-unacked task, with how long it has been outstanding.

    Returned by :meth:`~industryiq.core.agents.ports.TaskQueue.inflight` for
    observability and to drive reclaim: ``idle_ms`` is the time since the task was
    last delivered, ``delivery_count`` how many times it has been delivered (1 =
    never reclaimed). ``consumer`` is who currently holds it.
    """

    task: Task
    consumer: str
    idle_ms: float
    delivery_count: int


@dataclass(frozen=True)
class CapabilityResult:
    """The uniform envelope every capability returns.

    ``summary`` is the natural-language result a synthesizer reads; ``data`` is
    optional structured output; ``sources`` carries citations (each an opaque dict
    of doc/title/score) so the final answer stays grounded. Kept JSON-serializable
    so it can live on the blackboard verbatim -- see :meth:`as_dict`.
    """

    summary: str
    data: dict[str, Any] | None = None
    sources: list[dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        """Return a plain JSON-serializable dict (for the blackboard)."""
        return {"summary": self.summary, "data": self.data, "sources": self.sources}

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> CapabilityResult:
        """Rebuild from what :meth:`as_dict` wrote (tolerant of missing keys)."""
        return cls(
            summary=str(raw.get("summary", "")),
            data=raw.get("data"),
            sources=list(raw.get("sources", ())),
        )


@dataclass(frozen=True)
class PlanNode:
    """One subtask in a :class:`Plan`.

    Run ``capability`` on ``inputs`` (an opaque dict the capability understands),
    but only once every node named in ``depends_on`` has produced a result -- the
    DAG edges the supervisor/executor schedule on.
    """

    node_id: str
    capability: str
    inputs: dict[str, Any]
    depends_on: tuple[str, ...] = ()


@dataclass(frozen=True)
class Plan:
    """A decomposition of one request into a DAG of :class:`PlanNode` s."""

    run_id: str
    question: str
    nodes: tuple[PlanNode, ...]

    def node_ids(self) -> set[str]:
        """The set of all node ids in the plan."""
        return {node.node_id for node in self.nodes}

    def ready(self, done: set[str]) -> list[PlanNode]:
        """Nodes not yet in ``done`` whose dependencies are all in ``done``."""
        return [
            node
            for node in self.nodes
            if node.node_id not in done and all(dep in done for dep in node.depends_on)
        ]


@dataclass(frozen=True)
class RunResult:
    """The outcome of executing a :class:`Plan` (by either executor).

    ``answer`` is the synthesized, cited response; ``sources`` the merged citations;
    ``completed`` / ``failed`` name the nodes that did and did not produce a result
    (``failed`` is non-empty only for a partial run -- a dead-lettered or timed-out
    node).
    """

    run_id: str
    question: str
    answer: str
    sources: list[dict[str, Any]] = field(default_factory=list)
    completed: tuple[str, ...] = ()
    failed: tuple[str, ...] = ()
