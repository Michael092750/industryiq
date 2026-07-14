"""The composition root selects Redis-backed agent stores when REDIS_URL is set,
and the in-process doubles otherwise -- mirroring every other store's seam. All
offline: the Redis client is lazy, so selection opens no connection.
"""

import pytest

from industryiq.api import deps
from industryiq.core.agents.adapters.blackboard_memory import InMemoryBlackboard
from industryiq.core.agents.adapters.blackboard_redis import RedisBlackboard
from industryiq.core.agents.adapters.queue_memory import InMemoryTaskQueue
from industryiq.core.agents.adapters.queue_redis import RedisTaskQueue


def _clear() -> None:
    deps.get_redis.cache_clear()
    deps.get_blackboard.cache_clear()
    deps.get_task_queue.cache_clear()


def test_blackboard_is_in_memory_without_redis(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("REDIS_URL", raising=False)
    _clear()
    try:
        assert isinstance(deps.get_blackboard(), InMemoryBlackboard)
    finally:
        _clear()


def test_blackboard_uses_redis_when_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379/0")
    _clear()
    try:
        assert isinstance(deps.get_blackboard(), RedisBlackboard)
    finally:
        _clear()


def test_task_queue_is_in_memory_without_redis(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("REDIS_URL", raising=False)
    _clear()
    try:
        assert isinstance(deps.get_task_queue(), InMemoryTaskQueue)
    finally:
        _clear()


def test_task_queue_uses_redis_when_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379/0")
    _clear()
    try:
        assert isinstance(deps.get_task_queue(), RedisTaskQueue)
    finally:
        _clear()
