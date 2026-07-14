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
