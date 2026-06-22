"""A Milvus implementation of the VectorStore interface.

Satisfies the same :class:`~industryiq.core.vectorstore.VectorStore` protocol as
:class:`~industryiq.core.vectorstore.InMemoryVectorStore` and
:class:`~industryiq.core.pgvectorstore.PgVectorStore`, so it is a drop-in
replacement -- the pipeline and API are unchanged. Like pgvector, data persists
across restarts (in the Milvus standalone server).

Design choices that keep parity with the pgvector store, so the two can be
benchmarked against the same query/gold set:

* The primary key is a ``VARCHAR`` holding the caller's string ids (UUIDs), not a
  Milvus auto-id -- ``upsert`` is therefore a true replace-by-id.
* Metadata is one ``JSON`` field (mirrors pgvector's ``JSONB`` column) rather than
  a column per attribute, so arbitrary metadata round-trips unchanged.
* The metric is ``COSINE``, so Milvus returns cosine *similarity* directly as the
  hit ``distance`` -- numerically comparable to pgvector's ``1 - cosine_distance``.

Connections are managed by a single :class:`MilvusClient`; it is thread-safe for
the per-request use the web server makes of it.
"""

from typing import Any

from pymilvus import DataType, MilvusClient

from industryiq.core.vectorstore import Hit, VectorStore

# Milvus caps a single ``query`` at 16,384 rows; ``all_items`` pages with the
# query iterator instead, in batches of this size.
_ITER_BATCH = 1000

# Default *build-time* params per index type (override via ``index_params``). The
# index method is explicit (not AUTOINDEX) so benchmark runs are reproducible and
# self-documenting -- you know exactly which index produced a number.
_DEFAULT_INDEX_PARAMS: dict[str, dict[str, Any]] = {
    "HNSW": {"M": 16, "efConstruction": 200},
    "IVF_FLAT": {"nlist": 128},
    "IVF_SQ8": {"nlist": 128},
    "FLAT": {},
    "AUTOINDEX": {},
}
# Default *search-time* params. For HNSW ``ef`` is raised to at least k at search
# time (Milvus requires ef >= top_k); for IVF ``nprobe`` buckets are scanned.
_DEFAULT_EF = 128
_DEFAULT_NPROBE = 16


class MilvusVectorStore(VectorStore):
    """Vector store backed by a Milvus collection (standalone server).

    ``index_type`` selects the vector index explicitly (default ``HNSW``); pass
    ``index_params`` to override the build-time params for that index. To compare
    index methods, rebuild the collection with a different ``index_type`` (e.g.
    drop + re-migrate) and re-run the benchmark -- the metric (COSINE) and data
    are unchanged, so the difference is purely the index.
    """

    def __init__(
        self,
        uri: str,
        dim: int,
        collection: str = "chunks",
        token: str | None = None,
        *,
        index_type: str = "HNSW",
        index_params: dict[str, Any] | None = None,
    ) -> None:
        self._client = MilvusClient(uri=uri, token=token or "")
        self._collection = collection
        self._dim = dim
        self._index_type = index_type
        self._index_params = (
            index_params
            if index_params is not None
            else dict(_DEFAULT_INDEX_PARAMS.get(index_type, {}))
        )
        if not self._client.has_collection(collection):
            self._create_collection(collection, dim)
        # Idempotent; needed after a server restart even if the collection exists.
        self._client.load_collection(collection)

    def _create_collection(self, collection: str, dim: int) -> None:
        schema = self._client.create_schema(auto_id=False, enable_dynamic_field=False)
        # VARCHAR pk so the caller's UUIDs are the primary key (upsert = replace).
        schema.add_field("id", DataType.VARCHAR, is_primary=True, max_length=512)
        schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=dim)
        schema.add_field("metadata", DataType.JSON)

        # Explicit index method + params; COSINE makes the search ``distance`` a
        # cosine similarity (comparable to pgvector's ``1 - cosine_distance``).
        index_params = self._client.prepare_index_params()
        index_params.add_index(
            field_name="embedding",
            index_type=self._index_type,
            metric_type="COSINE",
            params=self._index_params,
        )
        # Strong consistency so a write is immediately visible to the next search
        # (read-your-writes), matching the pgvector store's behavior.
        self._client.create_collection(
            collection_name=collection,
            schema=schema,
            index_params=index_params,
            consistency_level="Strong",
        )

    def _search_params(self, k: int) -> dict[str, Any]:
        """Search-time params matched to the configured index method."""
        params: dict[str, Any] = {}
        if self._index_type == "HNSW":
            params = {"ef": max(_DEFAULT_EF, k)}  # Milvus requires ef >= top_k.
        elif self._index_type.startswith("IVF"):
            params = {"nprobe": _DEFAULT_NPROBE}
        return {"metric_type": "COSINE", "params": params}

    def upsert(
        self,
        ids: list[str],
        vectors: list[list[float]],
        metadatas: list[dict[str, Any]],
    ) -> None:
        if not (len(ids) == len(vectors) == len(metadatas)):
            raise ValueError("ids, vectors, and metadatas must have equal length")
        if not ids:
            return
        rows = [
            {"id": id_, "embedding": vector, "metadata": metadata}
            for id_, vector, metadata in zip(ids, vectors, metadatas, strict=True)
        ]
        self._client.upsert(collection_name=self._collection, data=rows)

    def search(self, query: list[float], k: int = 5) -> list[Hit]:
        if k <= 0:
            raise ValueError("k must be positive")
        results = self._client.search(
            collection_name=self._collection,
            data=[query],
            limit=k,
            output_fields=["metadata"],
            search_params=self._search_params(k),
        )
        if not results:
            return []
        return [
            Hit(
                id=hit["id"],
                score=float(hit["distance"]),
                metadata=hit["entity"].get("metadata", {}),
            )
            for hit in results[0]
        ]

    def all_items(self, limit: int = 100) -> list[tuple[str, dict[str, Any]]]:
        items: list[tuple[str, dict[str, Any]]] = []
        iterator = self._client.query_iterator(
            collection_name=self._collection,
            filter="",
            output_fields=["id", "metadata"],
            batch_size=min(_ITER_BATCH, limit),
        )
        try:
            while len(items) < limit:
                batch = iterator.next()
                if not batch:
                    break
                for row in batch:
                    items.append((row["id"], row.get("metadata", {})))
                    if len(items) >= limit:
                        break
        finally:
            iterator.close()
        return items
