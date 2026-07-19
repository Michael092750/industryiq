"""Redis-backed vector store for the hot / session tier.

One Redis hash per store instance (its ``key``): field = chunk id, value = a JSON
record ``{"vector": [...], "metadata": {...}}``. :meth:`search` loads the hash and
ranks by cosine similarity in Python -- so it needs no RediSearch module and runs
on stock Redis (the ``redis:7-alpine`` in docker-compose).

Deliberately for **bounded** namespaces -- a single conversation's uploaded
documents, tens or hundreds of chunks -- where brute-force ranking is trivial,
exactly as :class:`~industryiq.core.vectorstore.InMemoryVectorStore` already does
for the same job. It is NOT for the full corpus: that stays in pgvector/Milvus,
which index for sub-linear search. Here the whole win is that the data is *shared*
across processes and *survives a restart*, which an in-process dict cannot be.

An optional ``ttl_seconds`` makes the namespace self-evict: it is refreshed on
every :meth:`upsert`, so an idle session's docs expire while an active session's
persist. Deletion is explicit via :meth:`delete_by_source` (or dropping the whole
key from the owning store).

Mirrors :class:`InMemoryVectorStore`'s semantics method-for-method so it satisfies
the same :class:`~industryiq.core.vectorstore.VectorStore` contract and tests.
"""

import json
from typing import Any

from industryiq.core.vectorstore import Hit, VectorStore, cosine_similarity


class RedisVectorStore(VectorStore):
    """A single-hash, cosine-in-Python vector store for one bounded namespace.

    ``redis`` is a ``redis.Redis`` client built with ``decode_responses=True`` (see
    :func:`industryiq.core.redis_client.build_redis_client`). It is typed ``Any``
    because redis-py's stubs don't model ``decode_responses`` -- the whole project
    treats the client as untyped (see the mypy overrides in pyproject).
    """

    def __init__(self, redis: Any, key: str, *, ttl_seconds: int | None = None) -> None:
        self._redis = redis
        self._key = key
        self._ttl = ttl_seconds

    def upsert(
        self,
        ids: list[str],
        vectors: list[list[float]],
        metadatas: list[dict[str, Any]],
    ) -> None:
        if not (len(ids) == len(vectors) == len(metadatas)):
            raise ValueError("ids, vectors, and metadatas must have equal length")
        if not ids:  # HSET rejects an empty mapping; nothing to do anyway.
            return
        mapping = {
            id_: json.dumps({"vector": vector, "metadata": metadata})
            for id_, vector, metadata in zip(ids, vectors, metadatas, strict=True)
        }
        self._redis.hset(self._key, mapping=mapping)
        if self._ttl is not None:
            self._redis.expire(self._key, self._ttl)

    def search(self, query: list[float], k: int = 5, *, query_text: str | None = None) -> list[Hit]:
        # Dense-only store: query_text is accepted for protocol parity but unused.
        if k <= 0:
            raise ValueError("k must be positive")
        raw: dict[str, str] = self._redis.hgetall(self._key)
        hits = [
            Hit(
                id=id_,
                score=cosine_similarity(query, record["vector"]),
                metadata=record["metadata"],
            )
            for id_, record in ((id_, json.loads(blob)) for id_, blob in raw.items())
        ]
        hits.sort(key=lambda hit: hit.score, reverse=True)
        return hits[:k]

    def all_items(self, limit: int = 100) -> list[tuple[str, dict[str, Any]]]:
        raw: dict[str, str] = self._redis.hgetall(self._key)
        items = [(id_, json.loads(blob)["metadata"]) for id_, blob in raw.items()]
        return items[:limit]

    def delete_by_source(self, source: str) -> int:
        raw: dict[str, str] = self._redis.hgetall(self._key)
        ids = [
            id_ for id_, blob in raw.items() if json.loads(blob)["metadata"].get("source") == source
        ]
        if ids:
            self._redis.hdel(self._key, *ids)
        return len(ids)

    def fetch_neighbors(self, source: str, indices: list[int]) -> dict[int, dict[str, Any]]:
        # Brute-force scan of the bounded namespace, keyed by chunk_index (mirrors
        # InMemoryVectorStore). Cheap here since a session holds tens of chunks.
        wanted = set(indices)
        raw: dict[str, str] = self._redis.hgetall(self._key)
        result: dict[int, dict[str, Any]] = {}
        for blob in raw.values():
            metadata = json.loads(blob)["metadata"]
            index = metadata.get("chunk_index")
            if metadata.get("source") == source and index in wanted:
                result[int(index)] = metadata
        return result
