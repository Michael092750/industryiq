"""Integration test: a real PING against a live Redis.

Run with Redis available (e.g. `docker compose up -d redis`) via:

    pytest -m integration

Skipped from the default unit run. Proves the onboarding wiring end to end:
build a client from REDIS_URL, and the server answers.
"""

import pytest

from industryiq.config import get_settings
from industryiq.core.redis_client import build_redis_client, ping

pytestmark = pytest.mark.integration

REDIS_URL = get_settings().redis_url


def test_real_ping() -> None:
    if not REDIS_URL:
        pytest.skip("REDIS_URL not set")
    client = build_redis_client(REDIS_URL)
    try:
        assert ping(client) is True
    finally:
        client.close()
