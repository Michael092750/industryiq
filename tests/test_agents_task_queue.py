"""Contract tests for the TaskQueue port.

Each test runs twice -- against the in-memory double and the Redis Streams adapter
(on fakeredis) -- so both honor one behavior spec: competing consumers, at-least-
once delivery, and ack-to-complete. Live Redis is exercised in
test_agents_task_queue_integration.py.
"""

import fakeredis
import pytest

from industryiq.core.agents.adapters.queue_memory import InMemoryTaskQueue
from industryiq.core.agents.adapters.queue_redis import RedisTaskQueue
from industryiq.core.agents.ports import TaskQueue


@pytest.fixture(params=["memory", "redis"])
def queue(request: pytest.FixtureRequest) -> TaskQueue:
    if request.param == "memory":
        return InMemoryTaskQueue()
    return RedisTaskQueue(fakeredis.FakeStrictRedis(decode_responses=True))


def test_enqueue_increments_pending(queue: TaskQueue) -> None:
    queue.enqueue("q", {"n": 1})
    queue.enqueue("q", {"n": 2})
    assert queue.pending("q") == 2


def test_claim_returns_task_with_payload_and_id(queue: TaskQueue) -> None:
    queue.enqueue("q", {"n": 7})
    tasks = queue.claim("q", "w1")
    assert len(tasks) == 1
    assert tasks[0].payload == {"n": 7}
    assert tasks[0].id


def test_claim_empty_queue_returns_empty(queue: TaskQueue) -> None:
    assert queue.claim("q", "w1") == []


def test_competing_consumers_get_distinct_tasks(queue: TaskQueue) -> None:
    queue.enqueue("q", {"n": 1})
    queue.enqueue("q", {"n": 2})
    first = queue.claim("q", "w1", count=1)
    second = queue.claim("q", "w2", count=1)
    assert first[0].id != second[0].id  # never delivered twice


def test_claim_count_caps_the_batch(queue: TaskQueue) -> None:
    for i in range(3):
        queue.enqueue("q", {"n": i})
    assert len(queue.claim("q", "w1", count=2)) == 2


def test_claimed_task_stays_outstanding_until_ack(queue: TaskQueue) -> None:
    queue.enqueue("q", {"n": 1})
    [task] = queue.claim("q", "w1")
    assert queue.pending("q") == 1  # claimed but not acked
    queue.ack("q", task)
    assert queue.pending("q") == 0


def test_claimed_task_is_not_redelivered_before_ack(queue: TaskQueue) -> None:
    queue.enqueue("q", {"n": 1})
    queue.claim("q", "w1")
    assert queue.claim("q", "w2") == []  # the only task is in-flight, not redelivered


def test_queues_are_isolated(queue: TaskQueue) -> None:
    queue.enqueue("q1", {"n": 1})
    assert queue.pending("q2") == 0
    assert queue.claim("q2", "w1") == []


# --- recovery: inflight / reclaim / dead-letter (contract, both adapters) ---------


def test_inflight_reports_claimed_tasks(queue: TaskQueue) -> None:
    queue.enqueue("q", {"n": 1})
    queue.enqueue("q", {"n": 2})
    queue.claim("q", "w1", count=2)
    in_flight = queue.inflight("q")
    assert len(in_flight) == 2
    assert {f.consumer for f in in_flight} == {"w1"}
    assert {f.delivery_count for f in in_flight} == {1}


def test_inflight_empty_when_nothing_claimed(queue: TaskQueue) -> None:
    queue.enqueue("q", {"n": 1})  # undelivered, not claimed
    assert queue.inflight("q") == []


def test_reclaim_redelivers_an_abandoned_task(queue: TaskQueue) -> None:
    queue.enqueue("q", {"n": 1})
    queue.claim("q", "w1")  # w1 claims, then "dies" without acking
    reclaimed = queue.reclaim("q", "w2", min_idle_ms=0)
    assert len(reclaimed) == 1
    assert reclaimed[0].payload == {"n": 1}
    assert reclaimed[0].attempt == 2  # delivered once to w1, now again to w2
    queue.ack("q", reclaimed[0])
    assert queue.pending("q") == 0


def test_reclaim_ignores_undelivered_backlog(queue: TaskQueue) -> None:
    queue.enqueue("q", {"n": 1})  # never claimed -> not outstanding -> not reclaimable
    assert queue.reclaim("q", "w2", min_idle_ms=0) == []


def test_reclaim_empty_queue_returns_empty(queue: TaskQueue) -> None:
    assert queue.reclaim("q", "w1", min_idle_ms=0) == []


def test_dead_letter_moves_task_off_the_queue(queue: TaskQueue) -> None:
    queue.enqueue("q", {"n": 1})
    [task] = queue.claim("q", "w1")
    queue.dead_letter("q", task, reason="boom")
    assert queue.pending("q") == 0
    dead = queue.dead("q")
    assert len(dead) == 1
    assert dead[0].payload == {"n": 1}


# --- recovery: idle timing (in-memory double, deterministic via injected clock) ---


class _Clock:
    """A hand-cranked monotonic clock so idle-time is deterministic, no sleeps."""

    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += seconds


def test_reclaim_respects_min_idle_ms() -> None:
    clock = _Clock()
    queue = InMemoryTaskQueue(clock=clock)
    queue.enqueue("q", {"n": 1})
    queue.claim("q", "w1")
    clock.advance(0.5)  # 500 ms idle
    assert queue.reclaim("q", "w2", min_idle_ms=1000) == []  # not idle enough
    clock.advance(1.0)  # now 1500 ms idle
    reclaimed = queue.reclaim("q", "w2", min_idle_ms=1000)
    assert len(reclaimed) == 1
    assert reclaimed[0].attempt == 2


def test_reclaim_increments_attempt_each_time() -> None:
    clock = _Clock()
    queue = InMemoryTaskQueue(clock=clock)
    queue.enqueue("q", {"n": 1})
    queue.claim("q", "w1")
    clock.advance(2.0)
    first = queue.reclaim("q", "w2", min_idle_ms=1000)
    assert first[0].attempt == 2
    clock.advance(2.0)
    second = queue.reclaim("q", "w3", min_idle_ms=1000)
    assert second[0].attempt == 3


def test_inflight_reports_idle_ms_from_clock() -> None:
    clock = _Clock()
    queue = InMemoryTaskQueue(clock=clock)
    queue.enqueue("q", {"n": 1})
    queue.claim("q", "w1")
    clock.advance(1.5)
    [in_flight] = queue.inflight("q")
    assert in_flight.idle_ms == pytest.approx(1500.0)
