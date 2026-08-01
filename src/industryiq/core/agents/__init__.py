"""Multi-agent coordination substrate: a supervisor + workers shaped as ports-and-adapters.

Public surface:

* Ports (:mod:`industryiq.core.agents.ports`) -- :class:`Blackboard` (shared
  working memory), :class:`TaskQueue` (competing-consumers work distribution with
  reclaim), :class:`Capability`, :class:`Planner`, and :class:`RunLedger`.
* Models (:mod:`industryiq.core.agents.models`) -- :class:`Task`, :class:`InFlight`,
  :class:`CapabilityResult`, :class:`PlanNode`, :class:`Plan`.
* In-memory adapters -- the defaults/test doubles. The Redis adapters (and the
  heavier capability/planner/executor modules) are imported only where they are
  wired (:mod:`industryiq.api.deps`), to keep this package import light (no redis
  import at package load).
"""

from industryiq.core.agents.adapters.blackboard_memory import InMemoryBlackboard
from industryiq.core.agents.adapters.ledger_memory import InMemoryRunLedger
from industryiq.core.agents.adapters.queue_memory import InMemoryTaskQueue
from industryiq.core.agents.models import (
    CapabilityResult,
    InFlight,
    Plan,
    PlanNode,
    Task,
)
from industryiq.core.agents.ports import (
    Blackboard,
    Capability,
    Planner,
    RunLedger,
    TaskQueue,
)

__all__ = [
    "Blackboard",
    "Capability",
    "CapabilityResult",
    "InFlight",
    "InMemoryBlackboard",
    "InMemoryRunLedger",
    "InMemoryTaskQueue",
    "Plan",
    "PlanNode",
    "Planner",
    "RunLedger",
    "Task",
    "TaskQueue",
]
