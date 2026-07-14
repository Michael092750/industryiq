"""Integration test: RedisTaskQueue (Redis Streams) against a real Redis.

Run with Redis available (`docker compose up -d redis`) via: pytest -m integration
fakeredis covers the logic; this proves the streams/consumer-group wiring against a
real server (enqueue -> two workers claim distinct tasks -> ack -> drained).
"""

import uuid

import pytest

from industryiq.config import get_settings
from industryiq.core.agents.adapters.queue_redis import RedisTaskQueue
from industryiq.core.redis_client import build_redis_client

pytestmark = pytest.mark.integration

REDIS_URL = get_settings().redis_url


def test_supervisor_worker_round_trip() -> None:
    if not REDIS_URL:
        pytest.skip("REDIS_URL not set")
    client = build_redis_client(REDIS_URL)
    queue = RedisTaskQueue(client)
    q = "itest-q-" + uuid.uuid4().hex[:8]
    try:
        queue.enqueue(q, {"sub": "one"})
        queue.enqueue(q, {"sub": "two"})
        assert queue.pending(q) == 2
        first = queue.claim(q, "w1", count=1)
        second = queue.claim(q, "w2", count=1)
        assert {first[0].id, second[0].id} == {first[0].id, second[0].id}
        assert first[0].id != second[0].id
        queue.ack(q, first[0])
        queue.ack(q, second[0])
        assert queue.pending(q) == 0
    finally:
        client.delete("taskq:" + q)
        client.close()
