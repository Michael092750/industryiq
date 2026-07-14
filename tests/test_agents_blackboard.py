"""Contract tests for the Blackboard port.

Each test runs twice -- against the in-memory double and the Redis adapter (on
fakeredis) -- so both honor one behavior spec. A live-Redis smoke test lives in
test_agents_blackboard_integration.py.
"""

import fakeredis
import pytest

from industryiq.core.agents.adapters.blackboard_memory import InMemoryBlackboard
from industryiq.core.agents.adapters.blackboard_redis import RedisBlackboard
from industryiq.core.agents.ports import Blackboard


@pytest.fixture(params=["memory", "redis"])
def blackboard(request: pytest.FixtureRequest) -> Blackboard:
    if request.param == "memory":
        return InMemoryBlackboard()
    return RedisBlackboard(fakeredis.FakeStrictRedis(decode_responses=True))


def test_write_then_read(blackboard: Blackboard) -> None:
    blackboard.write("run1", "plan", {"steps": 3})
    assert blackboard.read("run1", "plan") == {"steps": 3}


def test_read_absent_key_is_none(blackboard: Blackboard) -> None:
    assert blackboard.read("run1", "nope") is None


def test_entries_returns_full_namespace_snapshot(blackboard: Blackboard) -> None:
    blackboard.write("run1", "a", 1)
    blackboard.write("run1", "b", "two")
    assert blackboard.entries("run1") == {"a": 1, "b": "two"}


def test_entries_of_empty_namespace_is_empty(blackboard: Blackboard) -> None:
    assert blackboard.entries("empty") == {}


def test_write_overwrites_existing_key(blackboard: Blackboard) -> None:
    blackboard.write("run1", "k", "old")
    blackboard.write("run1", "k", "new")
    assert blackboard.read("run1", "k") == "new"


def test_delete_removes_key(blackboard: Blackboard) -> None:
    blackboard.write("run1", "k", 1)
    blackboard.delete("run1", "k")
    assert blackboard.read("run1", "k") is None


def test_delete_absent_key_is_noop(blackboard: Blackboard) -> None:
    blackboard.delete("run1", "ghost")  # must not raise


def test_clear_drops_the_namespace(blackboard: Blackboard) -> None:
    blackboard.write("run1", "a", 1)
    blackboard.clear("run1")
    assert blackboard.entries("run1") == {}


def test_namespaces_are_isolated(blackboard: Blackboard) -> None:
    blackboard.write("run1", "k", "one")
    blackboard.write("run2", "k", "two")
    assert blackboard.read("run1", "k") == "one"
    assert blackboard.read("run2", "k") == "two"
    blackboard.clear("run1")
    assert blackboard.read("run2", "k") == "two"


def test_redis_blackboard_sets_ttl_on_write() -> None:
    client = fakeredis.FakeStrictRedis(decode_responses=True)
    RedisBlackboard(client, ttl_seconds=100).write("run1", "k", 1)
    assert 0 < client.ttl("bb:run1") <= 100
