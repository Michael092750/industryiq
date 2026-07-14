"""Ports for the multi-agent coordination substrate.

Two abstractions, shaped for a supervisor + workers topology:

* :class:`Blackboard` -- shared working memory. A namespaced, structured, TTL'd
  key/value space every agent can read and write, so a worker can post a result
  and the supervisor can read it back. This is the *hot context* tier (Redis when
  configured), distinct from the durable corpus (pgvector/Milvus) and the
  conversation log (Postgres).

* :class:`TaskQueue` -- work distribution. The supervisor enqueues tasks; workers
  claim and ack them, each task delivered to exactly one worker (competing
  consumers, not fan-out). Delivery is at-least-once: a claimed task stays
  outstanding until acked.

Following Dependency Inversion, agents will depend on these ``Protocol`` s, never
on a concrete adapter -- an in-memory double in tests, Redis in production, and the
two are indistinguishable to the caller. Adapters declare the port as an explicit
base class so the abstraction-to-implementation link is checked at the definition
site.

Values written to the blackboard and task payloads must be JSON-serializable; they
are typed ``Any`` because that is the honest bound (any JSON value).
"""

from typing import Any, Protocol, runtime_checkable

from industryiq.core.agents.models import Task


@runtime_checkable
class Blackboard(Protocol):
    """Shared, namespaced working memory for agents.

    A ``namespace`` scopes one coordination context -- typically a single run or
    task id -- so concurrent runs never see each other's state. Within a namespace
    it is a plain key -> JSON-value map.
    """

    def write(self, namespace: str, key: str, value: Any) -> None:
        """Set ``key`` in ``namespace`` to ``value`` (JSON-serializable)."""
        ...

    def read(self, namespace: str, key: str) -> Any | None:
        """Return the value of ``key`` in ``namespace``, or ``None`` if absent."""
        ...

    def entries(self, namespace: str) -> dict[str, Any]:
        """Return the whole namespace as a ``{key: value}`` snapshot (``{}`` if empty)."""
        ...

    def delete(self, namespace: str, key: str) -> None:
        """Remove ``key`` from ``namespace`` (a no-op if it is not present)."""
        ...

    def clear(self, namespace: str) -> None:
        """Drop an entire namespace (e.g. when a run finishes)."""
        ...


@runtime_checkable
class TaskQueue(Protocol):
    """Competing-consumers work queue: enqueue tasks, claim, then ack.

    Each task is delivered to exactly one consumer of a ``queue``. A claimed task
    is *outstanding* (at-least-once) until :meth:`ack`; the in-memory double mirrors
    this without the crash-recovery reclaim a real broker adds.
    """

    def enqueue(self, queue: str, payload: dict[str, Any]) -> Task:
        """Add a task to ``queue`` and return it with its assigned id."""
        ...

    def claim(self, queue: str, consumer: str, *, count: int = 1) -> list[Task]:
        """Claim up to ``count`` undelivered tasks for ``consumer`` (may be empty)."""
        ...

    def ack(self, queue: str, task: Task) -> None:
        """Mark a claimed ``task`` complete, removing it from ``queue``."""
        ...

    def pending(self, queue: str) -> int:
        """Return how many tasks are still outstanding (unacked) in ``queue``."""
        ...
