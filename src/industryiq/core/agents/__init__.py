"""Multi-agent coordination substrate: a supervisor + workers shaped as ports-and-adapters.

Public surface:

* Ports (:mod:`industryiq.core.agents.ports`) -- :class:`Blackboard` (shared
  working memory) and :class:`TaskQueue` (competing-consumers work distribution).
* Models (:mod:`industryiq.core.agents.models`) -- :class:`Task`.
* In-memory adapters -- the defaults/test doubles. The Redis adapters are imported
  only where they are wired (:mod:`industryiq.api.deps`), to keep this package
  import light (no redis import at package load).
"""

from industryiq.core.agents.adapters.blackboard_memory import InMemoryBlackboard
from industryiq.core.agents.adapters.queue_memory import InMemoryTaskQueue
from industryiq.core.agents.models import Task
from industryiq.core.agents.ports import Blackboard, TaskQueue

__all__ = [
    "Blackboard",
    "InMemoryBlackboard",
    "InMemoryTaskQueue",
    "Task",
    "TaskQueue",
]
