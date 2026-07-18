"""Integration test: RedisSessionDocumentStore against a real Redis.

Run with Redis available (e.g. `docker compose up -d redis`) via:

    pytest -m integration

Skipped from the default unit run. fakeredis covers the logic in the unit suite;
this proves the wiring works end to end against a real server (add -> retrieve ->
documents -> clear), on a throwaway conversation id.
"""

import uuid

import pytest

from industryiq.config import get_settings
from industryiq.core.embeddings import FakeEmbedder
from industryiq.core.redis_client import build_redis_client
from industryiq.core.retrieval.adapters.session_documents_redis import RedisSessionDocumentStore

pytestmark = pytest.mark.integration

REDIS_URL = get_settings().redis_url


def test_add_retrieve_documents_clear_round_trip() -> None:
    if not REDIS_URL:
        pytest.skip("REDIS_URL not set")
    client = build_redis_client(REDIS_URL)
    docs = RedisSessionDocumentStore(client, FakeEmbedder(dim=16), ttl_seconds=60)
    cid = "itest-" + uuid.uuid4().hex[:8]
    try:
        docs.add(cid, "facts.txt", "the sky is blue")
        hits = docs.retrieve(cid, "the sky is blue", k=3)
        assert hits and hits[0].metadata["source"] == "facts.txt"
        assert docs.documents(cid) == ["facts.txt"]
        docs.clear(cid)
        assert docs.documents(cid) == []
    finally:
        docs.clear(cid)
        client.close()
