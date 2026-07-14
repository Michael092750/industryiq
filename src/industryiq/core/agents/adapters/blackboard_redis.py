"""Redis-backed blackboard: shared across processes, survives restart, TTL'd.

One Redis hash per namespace (``bb:{namespace}``): field = key, value = the
JSON-encoded value. This is the multi-agent counterpart of
:class:`~industryiq.core.redisvectorstore.RedisVectorStore` -- same storage shape
(a hash of JSON), different job (arbitrary working memory, not vectors).

An optional ``ttl_seconds`` makes a namespace self-evict, refreshed on every
:meth:`write`, so a finished/abandoned run's scratch state does not accumulate.

``redis`` is a ``redis.Redis`` client (``decode_responses=True``), typed ``Any``
because redis-py's stubs don't model ``decode_responses`` -- the project treats the
client as untyped (see the mypy overrides in pyproject).
"""

import json
from typing import Any

from industryiq.core.agents.ports import Blackboard

_KEY_PREFIX = "bb:"


class RedisBlackboard(Blackboard):
    """Shared agent working memory backed by one Redis hash per namespace."""

    def __init__(self, redis: Any, *, ttl_seconds: int | None = None) -> None:
        self._redis = redis
        self._ttl = ttl_seconds

    def _key(self, namespace: str) -> str:
        return f"{_KEY_PREFIX}{namespace}"

    def write(self, namespace: str, key: str, value: Any) -> None:
        redis_key = self._key(namespace)
        self._redis.hset(redis_key, key, json.dumps(value))
        if self._ttl is not None:
            self._redis.expire(redis_key, self._ttl)

    def read(self, namespace: str, key: str) -> Any | None:
        blob = self._redis.hget(self._key(namespace), key)
        return json.loads(blob) if blob is not None else None

    def entries(self, namespace: str) -> dict[str, Any]:
        raw: dict[str, str] = self._redis.hgetall(self._key(namespace))
        return {key: json.loads(blob) for key, blob in raw.items()}

    def delete(self, namespace: str, key: str) -> None:
        self._redis.hdel(self._key(namespace), key)

    def clear(self, namespace: str) -> None:
        self._redis.delete(self._key(namespace))
