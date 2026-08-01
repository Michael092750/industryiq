"""Contract tests for the RunLedger port.

Each test runs against the in-memory double and the Redis Streams adapter (on
fakeredis), so both honor one spec: append-only, ordered, per-run isolation.
"""

import fakeredis
import pytest

from industryiq.core.agents.adapters.ledger_memory import InMemoryRunLedger
from industryiq.core.agents.adapters.ledger_redis import RedisRunLedger
from industryiq.core.agents.ports import RunLedger


@pytest.fixture(params=["memory", "redis"])
def ledger(request: pytest.FixtureRequest) -> RunLedger:
    if request.param == "memory":
        return InMemoryRunLedger()
    return RedisRunLedger(fakeredis.FakeStrictRedis(decode_responses=True))


def test_events_are_returned_in_append_order(ledger: RunLedger) -> None:
    ledger.append("r1", {"event": "plan_created"})
    ledger.append("r1", {"event": "task_enqueued", "node_id": "n1"})
    ledger.append("r1", {"event": "result_written", "node_id": "n1"})
    events = ledger.events("r1")
    assert [e["event"] for e in events] == ["plan_created", "task_enqueued", "result_written"]


def test_events_preserve_nested_payloads(ledger: RunLedger) -> None:
    ledger.append("r1", {"event": "plan_created", "nodes": [{"node_id": "n1", "depends_on": []}]})
    [event] = ledger.events("r1")
    assert event["nodes"][0]["node_id"] == "n1"


def test_runs_are_isolated(ledger: RunLedger) -> None:
    ledger.append("r1", {"event": "a"})
    ledger.append("r2", {"event": "b"})
    assert [e["event"] for e in ledger.events("r1")] == ["a"]
    assert [e["event"] for e in ledger.events("r2")] == ["b"]


def test_unknown_run_has_no_events(ledger: RunLedger) -> None:
    assert ledger.events("nope") == []
