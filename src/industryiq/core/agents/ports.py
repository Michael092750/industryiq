"""Ports for the multi-agent coordination substrate.

The abstractions a supervisor + workers topology depends on:

* :class:`Blackboard` -- shared working memory. A namespaced, structured, TTL'd
  key/value space every agent can read and write, so a worker can post a result
  and the supervisor can read it back. This is the *hot context* tier (Redis when
  configured), distinct from the durable corpus (pgvector/Milvus) and the
  conversation log (Postgres).

* :class:`TaskQueue` -- work distribution. The supervisor enqueues tasks; workers
  claim and ack them, each task delivered to exactly one worker (competing
  consumers, not fan-out). Delivery is at-least-once: a claimed task stays
  outstanding until acked, and :meth:`TaskQueue.reclaim` re-delivers one a crashed
  worker abandoned -- the recovery a distributed run needs.

* :class:`Capability` -- a named unit of work an agent can dispatch to (retrieval
  now; web search / database lookup later), all returning one uniform
  :class:`~industryiq.core.agents.models.CapabilityResult`.

* :class:`Planner` -- decomposes a question into a :class:`Plan` over the
  available capabilities.

* :class:`RunLedger` -- an append-only event log per run, so a distributed run
  has a single reconstructable timeline.

Following Dependency Inversion, agents depend on these ``Protocol`` s, never on a
concrete adapter -- an in-memory double in tests, Redis in production, and the two
are indistinguishable to the caller. Adapters declare the port as an explicit base
class so the abstraction-to-implementation link is checked at the definition site.

Values written to the blackboard and task payloads must be JSON-serializable; they
are typed ``Any`` because that is the honest bound (any JSON value).
"""

from collections.abc import Mapping
from typing import Any, Protocol, runtime_checkable

from industryiq.core.agents.models import CapabilityResult, InFlight, Plan, Task


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
    """Competing-consumers work queue: enqueue tasks, claim, ack -- and reclaim.

    Each task is delivered to exactly one consumer of a ``queue``. A claimed task
    is *outstanding* (at-least-once) until :meth:`ack`. If the consumer that holds
    it dies, :meth:`reclaim` re-delivers it to a live worker once it has been idle
    long enough; a task that keeps failing can be moved aside with
    :meth:`dead_letter`.
    """

    def enqueue(self, queue: str, payload: dict[str, Any]) -> Task:
        """Add a task to ``queue`` and return it with its assigned id."""
        ...

    def claim(self, queue: str, consumer: str, *, count: int = 1) -> list[Task]:
        """Claim up to ``count`` never-delivered tasks for ``consumer`` (may be empty)."""
        ...

    def ack(self, queue: str, task: Task) -> None:
        """Mark a claimed ``task`` complete, removing it from ``queue``."""
        ...

    def pending(self, queue: str) -> int:
        """Return how many tasks are still outstanding (unacked) in ``queue``."""
        ...

    def inflight(self, queue: str) -> list[InFlight]:
        """Return the claimed-but-unacked tasks in ``queue`` (empty if none).

        Separates true in-progress work (and how long each has been idle) from the
        undelivered backlog that :meth:`pending` lumps together -- the signal that
        surfaces a slow-vs-dead worker.
        """
        ...

    def reclaim(
        self, queue: str, consumer: str, *, min_idle_ms: float, count: int = 10
    ) -> list[Task]:
        """Re-deliver to ``consumer`` up to ``count`` tasks left unacked for at
        least ``min_idle_ms`` by some (possibly crashed) consumer.

        Each returned task's :attr:`~industryiq.core.agents.models.Task.attempt`
        reflects its incremented delivery count, so the caller can dead-letter a
        task that has been reclaimed too many times.
        """
        ...

    def dead_letter(self, queue: str, task: Task, *, reason: str) -> None:
        """Move ``task`` out of ``queue`` into its dead-letter store with ``reason``."""
        ...

    def dead(self, queue: str) -> list[Task]:
        """Return the dead-lettered tasks for ``queue`` (empty if none)."""
        ...


@runtime_checkable
class Capability(Protocol):
    """A named unit of work an agent can dispatch to.

    ``description`` is *prescriptive* ("use this when ...") because the planner
    routes on it; ``run`` reads an opaque JSON ``inputs`` dict and returns a
    uniform :class:`~industryiq.core.agents.models.CapabilityResult`. This seam is
    what lets retrieval, web search, and a database lookup all be dispatched the
    same way.

    ``name`` is the stable identifier a :class:`PlanNode` names; ``description`` is
    the one-line "use this when ..." blurb the planner routes on.
    """

    name: str
    description: str

    def run(self, inputs: dict[str, Any]) -> CapabilityResult:
        """Execute the capability on ``inputs`` and return its result envelope."""
        ...


@runtime_checkable
class Planner(Protocol):
    """Decompose a question into a :class:`Plan` over the available capabilities.

    ``capabilities`` maps capability name -> description; the planner routes on the
    descriptions alone, so it never depends on the concrete capability objects.
    The caller supplies ``run_id`` so the plan is tagged with the run it belongs to.
    """

    def plan(self, run_id: str, question: str, capabilities: Mapping[str, str]) -> Plan:
        """Return a :class:`Plan` (a DAG of subtasks) for ``question``."""
        ...


@runtime_checkable
class RunLedger(Protocol):
    """Append-only event log per run -- the reconstructable timeline for one run.

    A distributed run scatters its work across processes; the ledger is the one
    place its narrative (enqueued / claimed / reclaimed / result / done) is written
    in order, keyed by ``run_id``.
    """

    def append(self, run_id: str, event: dict[str, Any]) -> None:
        """Append ``event`` (a JSON-serializable dict) to ``run_id``'s log."""
        ...

    def events(self, run_id: str) -> list[dict[str, Any]]:
        """Return ``run_id``'s events in append order (``[]`` if none)."""
        ...
