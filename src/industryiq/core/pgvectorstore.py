"""A Postgres + pgvector implementation of the VectorStore interface.

Satisfies the same :class:`~industryiq.core.vectorstore.VectorStore` protocol as
:class:`~industryiq.core.vectorstore.InMemoryVectorStore`, so it is a drop-in
replacement -- the pipeline and API are unchanged. Unlike the in-memory store,
data persists in the database across restarts.

Connections are opened per operation for thread-safety under the web server.
"""

from typing import Any

import psycopg
from pgvector import Vector
from pgvector.psycopg import register_vector
from psycopg.types.json import Jsonb

from industryiq.core.vectorstore import Hit, VectorStore


class PgVectorStore(VectorStore):
    """Vector store backed by a Postgres table with a pgvector column."""

    def __init__(self, dsn: str, dim: int, table: str = "chunks") -> None:
        self._dsn = dsn
        self._dim = dim
        self._table = table
        # Create the extension on a plain connection FIRST -- register_vector()
        # (used by _connect) needs the vector type to already exist.
        with psycopg.connect(dsn) as conn:
            conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            conn.commit()
        with self._connect() as conn:
            conn.execute(
                f"CREATE TABLE IF NOT EXISTS {table} ("
                f"id TEXT PRIMARY KEY, embedding vector({dim}), metadata JSONB)"
            )
            conn.commit()

    def _connect(self) -> psycopg.Connection[Any]:
        conn = psycopg.connect(self._dsn)
        register_vector(conn)
        return conn

    def upsert(
        self,
        ids: list[str],
        vectors: list[list[float]],
        metadatas: list[dict[str, Any]],
    ) -> None:
        if not (len(ids) == len(vectors) == len(metadatas)):
            raise ValueError("ids, vectors, and metadatas must have equal length")
        with self._connect() as conn:
            for id_, vector, metadata in zip(ids, vectors, metadatas, strict=True):
                conn.execute(
                    f"INSERT INTO {self._table} (id, embedding, metadata) "
                    f"VALUES (%s, %s, %s) "
                    f"ON CONFLICT (id) DO UPDATE SET "
                    f"embedding = EXCLUDED.embedding, metadata = EXCLUDED.metadata",
                    (id_, Vector(vector), Jsonb(metadata)),
                )
            conn.commit()

    def search(self, query: list[float], k: int = 5) -> list[Hit]:
        if k <= 0:
            raise ValueError("k must be positive")
        with self._connect() as conn:
            query_vector = Vector(query)
            rows = conn.execute(
                f"SELECT id, metadata, 1 - (embedding <=> %s) AS score "
                f"FROM {self._table} ORDER BY embedding <=> %s LIMIT %s",
                (query_vector, query_vector, k),
            ).fetchall()
        return [Hit(id=row[0], score=float(row[2]), metadata=row[1]) for row in rows]

    def all_items(self, limit: int = 100) -> list[tuple[str, dict[str, Any]]]:
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT id, metadata FROM {self._table} LIMIT %s", (limit,)
            ).fetchall()
        return [(row[0], row[1]) for row in rows]
