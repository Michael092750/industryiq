"""In-memory blackboard: the default (no Redis) and the test double.

Dict-of-dicts, mirroring the other in-memory stores. It satisfies the
:class:`~industryiq.core.agents.ports.Blackboard` port, so agents cannot tell it
apart from the Redis-backed one -- except that this one is process-local and lost
on restart, which is exactly why Redis exists for real multi-process runs.
"""

from typing import Any

from industryiq.core.agents.ports import Blackboard


class InMemoryBlackboard(Blackboard):
    """A dict-backed blackboard for tests and single-process local development."""

    def __init__(self) -> None:
        self._spaces: dict[str, dict[str, Any]] = {}

    def write(self, namespace: str, key: str, value: Any) -> None:
        self._spaces.setdefault(namespace, {})[key] = value

    def read(self, namespace: str, key: str) -> Any | None:
        return self._spaces.get(namespace, {}).get(key)

    def entries(self, namespace: str) -> dict[str, Any]:
        return dict(self._spaces.get(namespace, {}))

    def delete(self, namespace: str, key: str) -> None:
        self._spaces.get(namespace, {}).pop(key, None)

    def clear(self, namespace: str) -> None:
        self._spaces.pop(namespace, None)
