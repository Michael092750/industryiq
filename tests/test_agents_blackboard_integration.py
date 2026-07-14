"""Integration test: RedisBlackboard against a real Redis.

Run with Redis available (`docker compose up -d redis`) via: pytest -m integration
fakeredis covers the logic; this proves the wiring against a real server.
"""

import uuid

import pytest

from industryiq.config import get_settings
from industryiq.core.agents.adapters.blackboard_redis import RedisBlackboard
from industryiq.core.redis_client import build_redis_client

pytestmark = pytest.mark.integration

REDIS_URL = get_settings().redis_url


def test_blackboard_round_trip() -> None:
    if not REDIS_URL:
        pytest.skip("REDIS_URL not set")
    client = build_redis_client(REDIS_URL)
    board = RedisBlackboard(client, ttl_seconds=60)
    ns = "itest-bb-" + uuid.uuid4().hex[:8]
    try:
        board.write(ns, "plan", {"steps": [1, 2, 3]})
        assert board.read(ns, "plan") == {"steps": [1, 2, 3]}
        board.write(ns, "done", True)
        assert board.entries(ns) == {"plan": {"steps": [1, 2, 3]}, "done": True}
        board.delete(ns, "done")
        assert board.read(ns, "done") is None
        board.clear(ns)
        assert board.entries(ns) == {}
    finally:
        board.clear(ns)
        client.close()
