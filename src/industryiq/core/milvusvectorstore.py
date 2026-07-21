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
* The frequently-searched metadata keys are promoted to their own typed columns
  (``text``, ``source``, ``section``, ``category``, ``publisher``,
  ``source_type``, ``published_date``) so they can be filtered server-side;
  everything else rides along in a residual ``metadata`` JSON field
  (mirroring pgvector's ``JSONB`` column). On read the columns are folded back
  into a single ``metadata`` dict, so the contract is still "dict in, same dict
  out" -- identical to pgvector -- and arbitrary keys (e.g. ``category``)
  round-trip unchanged.
* The dense metric is ``COSINE``, so a dense search returns cosine *similarity*
  directly as the hit ``distance`` -- numerically comparable to pgvector's
  ``1 - cosine_distance``.

Full-text (BM25)
----------------
The ``text`` column has an analyzer enabled and feeds a BM25 ``Function`` whose
output is the ``text_sparse`` ``SPARSE_FLOAT_VECTOR`` column, indexed with a
``SPARSE_INVERTED_INDEX``. Milvus generates the sparse term-weight vector at
insert (and the query's at search) -- callers never build it. When ``search`` is
given a ``query_text`` it runs a hybrid query (dense + BM25, RRF-fused for
ordering) and reports the dense cosine similarity as ``Hit.score``, so BM25
improves recall/ordering without changing the score scale; without ``query_text``
it is a plain dense search.

Connections are managed by a single :class:`MilvusClient`; it is thread-safe for
the per-request use the web server makes of it.
"""

import re
from typing import Any

from pymilvus import (
    AnnSearchRequest,
    DataType,
    Function,
    FunctionType,
    MilvusClient,
    RRFRanker,
    WeightedRanker,
)

from industryiq.core.vectorstore import (
    Hit,
    MetadataFilter,
    ScoreKind,
    SearchPlan,
    SearchStrategy,
    StrategicSearch,
    VectorStore,
    cosine_similarity,
)

# Milvus caps a single ``query`` at 16,384 rows; ``all_items`` pages with the
# query iterator instead, in batches of this size.
_ITER_BATCH = 1000

# Metadata keys promoted out of the JSON blob into their own typed VARCHAR columns
# so they can be searched/filtered server-side. Anything else stays in the
# residual ``metadata`` JSON field. All promoted keys are folded back into the
# metadata dict on read, so the read contract matches pgvector (the same dict
# that went in comes back out).
_TEXT_FIELD = "text"
_SOURCE_FIELD = "source"
_SECTION_FIELD = "section"
_CATEGORY_FIELD = "category"
_PUBLISHER_FIELD = "publisher"
_SOURCE_TYPE_FIELD = "source_type"
# ISO-8601 date string ("2024-03-15"); ISO sorts lexicographically = chronologically,
# so ``published_date >= "2024"`` works as a range filter on the VARCHAR column.
_PUBLISHED_DATE_FIELD = "published_date"
# Chunk position within its source document (INT64), promoted out of the JSON blob so
# neighbour/context expansion can filter it server-side (``source == X && chunk_index in [...]``).
_CHUNK_INDEX_FIELD = "chunk_index"
_METADATA_FIELD = "metadata"
_SCALAR_STR_FIELDS = (
    _TEXT_FIELD,
    _SOURCE_FIELD,
    _SECTION_FIELD,
    _CATEGORY_FIELD,
    _PUBLISHER_FIELD,
    _SOURCE_TYPE_FIELD,
    _PUBLISHED_DATE_FIELD,
)
# Promoted columns worth a scalar index because they are filtered/faceted on at
# query time (``text`` is excluded -- it is full-text indexed via BM25 instead).
_INDEXED_FIELDS = (
    _SOURCE_FIELD,
    _SECTION_FIELD,
    _CATEGORY_FIELD,
    _PUBLISHER_FIELD,
    _SOURCE_TYPE_FIELD,
    _PUBLISHED_DATE_FIELD,
)
# Every key promoted to its own column (str columns + the INT64 chunk_index); used to
# decide what stays behind in the residual JSON blob on write.
_PROMOTED_FIELDS = frozenset((*_SCALAR_STR_FIELDS, _CHUNK_INDEX_FIELD))
# Fields read back to reconstruct a hit's metadata dict.
_OUTPUT_FIELDS = (*_SCALAR_STR_FIELDS, _CHUNK_INDEX_FIELD, _METADATA_FIELD)
# Sentinel written when a chunk has no ``chunk_index`` (INT64 can't be absent); folded
# back to "no index" on read, so a chunk that went in without one comes back without one.
_NO_CHUNK_INDEX = -1

# BM25 full-text: ``text`` is analyzed and projected into a sparse term-weight
# vector by a Milvus Function; the sparse column is never set by callers.
_SPARSE_FIELD = "text_sparse"
_BM25_FUNCTION = "text_bm25"
# Standard analyzer: lowercase + alphanumeric tokenizer, no stopword removal, so
# discriminative tokens (digits, "Scope 3", acronyms) survive into the index.
_ANALYZER_PARAMS: dict[str, Any] = {"type": "standard"}
# BM25 build/search knobs. MAXSCORE prunes postings for top-k (sublinear); k1/b
# are the standard term-saturation and length-normalization parameters.
_BM25_INDEX_PARAMS: dict[str, Any] = {
    "inverted_index_algo": "DAAT_MAXSCORE",
    "bm25_k1": 1.2,
    "bm25_b": 0.75,
}

# VARCHAR capacities. ``text`` holds a whole chunk (~200 words) so it is sized
# near Milvus's 65,535-char VARCHAR ceiling; the rest are short.
_MAX_TEXT = 65535
_MAX_SOURCE = 2048
_MAX_SECTION = 1024
_MAX_CATEGORY = 256
_MAX_PUBLISHER = 512
_MAX_SOURCE_TYPE = 128
_MAX_PUBLISHED_DATE = 32
# Per-field VARCHAR length for the scalar string columns, by field name.
_SCALAR_STR_MAX = {
    _TEXT_FIELD: _MAX_TEXT,
    _SOURCE_FIELD: _MAX_SOURCE,
    _SECTION_FIELD: _MAX_SECTION,
    _CATEGORY_FIELD: _MAX_CATEGORY,
    _PUBLISHER_FIELD: _MAX_PUBLISHER,
    _SOURCE_TYPE_FIELD: _MAX_SOURCE_TYPE,
    _PUBLISHED_DATE_FIELD: _MAX_PUBLISHED_DATE,
}

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

# Hybrid (dense + BM25) search: each leg fetches this many candidates before
# fusion picks the final top-k. A pool larger than k improves fusion recall.
_HYBRID_CANDIDATES = 50
# RRF rank-fusion smoothing constant (standard default).
_RRF_K = 60
# Default weights for score-fusion (weighted) hybrid: dense (cosine) vs BM25.
_DEFAULT_DENSE_WEIGHT = 0.7
_DEFAULT_SPARSE_WEIGHT = 0.3


def _escape_literal(value: str) -> str:
    """Escape a string for safe interpolation into a Milvus filter string literal."""
    return value.replace("\\", "\\\\").replace('"', '\\"')


# A publisher filter value is reduced to this safe charset before a ``like`` match, so
# it can never carry a ``like`` wildcard (``%``/``_``) or a quote that breaks the literal.
_PUBLISHER_UNSAFE = re.compile(r"[^a-z0-9.\-]")


def _publisher_like_value(value: str) -> str:
    """Normalise a publisher filter value to a safe substring for a ``like`` match.

    The corpus tags ``publisher`` by lowercased domain, frequently with a subdomain
    (``www2.deloitte.com``, ``mkto.deloitte.com``). A *contains* match on the registrable
    domain (``deloitte.com``) therefore unifies all of a publisher's subdomain forms,
    and a plain name ("Deloitte") still matches via its token -- both far better than the
    exact ``==`` that missed every fragmented publisher. Lowercased, a leading ``www.``
    dropped, and restricted to ``[a-z0-9.-]``.
    """
    return _PUBLISHER_UNSAFE.sub("", value.strip().lower()).removeprefix("www.")


def _compile_filter_expr(mf: MetadataFilter) -> str:
    """Compile a :class:`MetadataFilter` into a Milvus boolean expression (pure).

    ``publisher`` is a case-insensitive substring (``like``) on the domain so subdomain
    variants and name/domain inputs all hit; ``source_type``/``category`` are exact
    equality on their controlled vocabularies; the dates are an inclusive range that
    *also admits undated chunks* (``published_date == ""``). Clauses are ANDed; an empty
    (or fully-unmatchable) filter compiles to ``""`` (no constraint).

    Undated chunks passing the date bound is deliberate: ``published_date`` is the
    *publication* year, ~37% of the corpus is undated, and a report about a year is
    routinely published in another (a 2025 report can cover 2023). Hard-excluding
    unknown/out-of-window publish dates silently drops docs that may hold the answer,
    so a date bound narrows *dated* chunks without ever dropping *undated* ones.
    """
    clauses: list[str] = []
    publisher = _publisher_like_value(mf.publisher) if mf.publisher else ""
    if publisher:
        clauses.append(f'{_PUBLISHER_FIELD} like "%{publisher}%"')
    if mf.source_type:
        clauses.append(f'{_SOURCE_TYPE_FIELD} == "{_escape_literal(mf.source_type)}"')
    if mf.category:
        clauses.append(f'{_CATEGORY_FIELD} == "{_escape_literal(mf.category)}"')
    date_bounds: list[str] = []
    if mf.published_from:
        date_bounds.append(f'{_PUBLISHED_DATE_FIELD} >= "{_escape_literal(mf.published_from)}"')
    if mf.published_to:
        date_bounds.append(f'{_PUBLISHED_DATE_FIELD} <= "{_escape_literal(mf.published_to)}"')
    if date_bounds:
        clauses.append(f'(({" && ".join(date_bounds)}) || {_PUBLISHED_DATE_FIELD} == "")')
    return " && ".join(clauses)


class MilvusVectorStore(VectorStore, StrategicSearch):
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
        # Promoted, server-searchable metadata columns. ``text`` additionally gets
        # an analyzer so it can feed the BM25 full-text function below.
        for field in _SCALAR_STR_FIELDS:
            extra: dict[str, Any] = (
                {"enable_analyzer": True, "analyzer_params": _ANALYZER_PARAMS}
                if field == _TEXT_FIELD
                else {}
            )
            schema.add_field(field, DataType.VARCHAR, max_length=_SCALAR_STR_MAX[field], **extra)
        # Promoted INT column: chunk position within its source doc, for neighbour
        # expansion. INT64 so it can be range/``in``-filtered server-side.
        schema.add_field(_CHUNK_INDEX_FIELD, DataType.INT64)
        # Residual metadata: anything not promoted above.
        schema.add_field(_METADATA_FIELD, DataType.JSON)
        # BM25 sparse column: Milvus fills it from ``text`` via the function below
        # (callers never set it), so it is excluded from upsert/output fields.
        schema.add_field(_SPARSE_FIELD, DataType.SPARSE_FLOAT_VECTOR)
        schema.add_function(
            Function(
                name=_BM25_FUNCTION,
                function_type=FunctionType.BM25,
                input_field_names=_TEXT_FIELD,
                output_field_names=_SPARSE_FIELD,
            )
        )

        # Explicit dense index method + params; COSINE makes the dense search
        # ``distance`` a cosine similarity (comparable to pgvector).
        index_params = self._client.prepare_index_params()
        index_params.add_index(
            field_name="embedding",
            index_type=self._index_type,
            metric_type="COSINE",
            params=self._index_params,
        )
        # BM25 sparse index for full-text/lexical search over ``text``.
        index_params.add_index(
            field_name=_SPARSE_FIELD,
            index_type="SPARSE_INVERTED_INDEX",
            metric_type="BM25",
            params=_BM25_INDEX_PARAMS,
        )
        # INVERTED scalar indexes make the promoted columns cheap to filter on
        # (``source``/``section``/``category`` equality).
        for field in _INDEXED_FIELDS:
            index_params.add_index(field_name=field, index_type="INVERTED")
        # chunk_index is filtered on (source + index range) by fetch_neighbors.
        index_params.add_index(field_name=_CHUNK_INDEX_FIELD, index_type="INVERTED")
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

    @staticmethod
    def _split_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
        """Map a caller metadata dict onto the promoted columns + residual JSON.

        Promoted keys become their own columns (with empty defaults when absent);
        everything else is kept verbatim in the ``metadata`` JSON field.
        """
        row: dict[str, Any] = {field: str(metadata.get(field, "")) for field in _SCALAR_STR_FIELDS}
        # chunk_index is an INT64 column; missing -> sentinel (INT64 can't be absent).
        raw_index = metadata.get(_CHUNK_INDEX_FIELD)
        row[_CHUNK_INDEX_FIELD] = int(raw_index) if raw_index is not None else _NO_CHUNK_INDEX
        row[_METADATA_FIELD] = {
            key: value for key, value in metadata.items() if key not in _PROMOTED_FIELDS
        }
        return row

    @staticmethod
    def _merge_metadata(row: dict[str, Any]) -> dict[str, Any]:
        """Fold the promoted columns back into one metadata dict (read side).

        Empty promoted values are treated as absent and omitted, so a chunk that
        went in as ``{"text": ...}`` comes back as exactly ``{"text": ...}`` --
        the same round-trip pgvector gives.
        """
        metadata = dict(row.get(_METADATA_FIELD) or {})
        for field in _SCALAR_STR_FIELDS:
            value = row.get(field)
            if value:
                metadata[field] = value
        # Fold chunk_index back only when it is a real position (not the sentinel).
        index = row.get(_CHUNK_INDEX_FIELD)
        if index is not None and index >= 0:
            metadata[_CHUNK_INDEX_FIELD] = int(index)
        return metadata

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
            {"id": id_, "embedding": vector, **self._split_metadata(metadata)}
            for id_, vector, metadata in zip(ids, vectors, metadatas, strict=True)
        ]
        self._client.upsert(collection_name=self._collection, data=rows)

    def search(self, query: list[float], k: int = 5, *, query_text: str | None = None) -> list[Hit]:
        """Return up to ``k`` hits, highest relevance first.

        With ``query_text`` the search is hybrid (dense + BM25, RRF-fused for
        ordering); without it, it is dense-only. Either way ``Hit.score`` is the
        dense cosine similarity, so scores stay comparable across stores and to
        pgvector -- BM25 only changes *which* chunks surface and their order.
        """
        if k <= 0:
            raise ValueError("k must be positive")
        if query_text:
            return self._hybrid_search(query, query_text, k)
        return self._dense_search(query, k)

    def search_plan(
        self,
        *,
        query_vector: list[float],
        query_text: str,
        k: int,
        plan: SearchPlan,
    ) -> list[Hit]:
        """Execute an explicit :class:`SearchPlan` -- the :class:`StrategicSearch` seam.

        Dispatches to the mode's primitive and threads the plan's metadata filter
        (compiled to a Milvus expression) into it. ``query_vector`` is embedded once
        upstream and reused; ``LEXICAL`` ignores it. Each mode stamps its own
        ``ScoreKind`` (via the underlying method), so the relevance filter downstream
        reads the score on the right scale.
        """
        if k <= 0:
            raise ValueError("k must be positive")
        expr = _compile_filter_expr(plan.filter) if plan.filter is not None else ""
        strategy = plan.strategy
        if strategy is SearchStrategy.SEMANTIC:
            return self._dense_search(query_vector, k, expr=expr)
        if strategy is SearchStrategy.LEXICAL:
            return self.lexical_search(query_text, k, expr=expr)
        if strategy is SearchStrategy.HYBRID_WEIGHTED:
            alpha, beta = plan.weights or (_DEFAULT_DENSE_WEIGHT, _DEFAULT_SPARSE_WEIGHT)
            return self.weighted_search(
                query_vector, query_text, k, alpha=alpha, beta=beta, expr=expr
            )
        # HYBRID_RRF: the default strategy, reachable here only when a filter is set.
        return self._hybrid_search(query_vector, query_text, k, expr=expr)

    def semantic_search(self, query: list[float], k: int = 5) -> list[Hit]:
        """Pure dense (vector) search; ``Hit.score`` is the cosine similarity.

        The semantic-only tool: best for conceptual/paraphrase queries where the
        exact wording doesn't matter. Equivalent to :meth:`search` without a
        ``query_text``, exposed under a name the router can select.
        """
        if k <= 0:
            raise ValueError("k must be positive")
        return self._dense_search(query, k)

    def _dense_search(self, query: list[float], k: int, *, expr: str = "") -> list[Hit]:
        results = self._client.search(
            collection_name=self._collection,
            data=[query],
            anns_field="embedding",
            limit=k,
            output_fields=list(_OUTPUT_FIELDS),
            search_params=self._search_params(k),
            filter=expr,
        )
        if not results:
            return []
        return [
            Hit(
                id=hit["id"],
                score=float(hit["distance"]),
                metadata=self._merge_metadata(hit["entity"]),
            )
            for hit in results[0]
        ]

    def _hybrid_search(
        self, query: list[float], query_text: str, k: int, *, expr: str = ""
    ) -> list[Hit]:
        pool = max(k, _HYBRID_CANDIDATES)
        dense = AnnSearchRequest(
            data=[query],
            anns_field="embedding",
            param=self._search_params(pool),
            limit=pool,
            expr=expr,
        )
        sparse = AnnSearchRequest(
            data=[query_text],
            anns_field=_SPARSE_FIELD,
            param={"metric_type": "BM25"},
            limit=pool,
            expr=expr,
        )
        fused = self._client.hybrid_search(
            collection_name=self._collection,
            reqs=[dense, sparse],
            ranker=RRFRanker(_RRF_K),
            limit=k,
            output_fields=["id"],
        )
        if not fused or not fused[0]:
            return []
        ordered_ids = [hit["id"] for hit in fused[0]]
        # Re-fetch the fused top-k so ``score`` can be the dense cosine similarity
        # (the RRF score is used only for ordering). k is small, so this by-id
        # get is cheap.
        rows = self._client.get(
            collection_name=self._collection,
            ids=ordered_ids,
            output_fields=["id", "embedding", *_OUTPUT_FIELDS],
        )
        by_id = {row["id"]: row for row in rows}
        hits: list[Hit] = []
        for id_ in ordered_ids:
            row = by_id.get(id_)
            if row is None:  # vanishingly rare (concurrent delete between calls)
                continue
            hits.append(
                Hit(
                    id=id_,
                    score=cosine_similarity(query, row["embedding"]),
                    metadata=self._merge_metadata(row),
                )
            )
        return hits

    def weighted_search(
        self,
        query: list[float],
        query_text: str,
        k: int = 5,
        *,
        alpha: float = _DEFAULT_DENSE_WEIGHT,
        beta: float = _DEFAULT_SPARSE_WEIGHT,
        expr: str = "",
    ) -> list[Hit]:
        """Score-fusion hybrid: ``score = alpha*norm(dense) + beta*norm(bm25)``.

        Unlike :meth:`search` (RRF order, raw cosine score), this reports the
        *blended, normalized* score and ranks by it -- bump ``beta`` for queries
        where exact terms/facts matter more than semantics. Milvus's
        ``WeightedRanker`` min-max normalizes each leg within the candidate set,
        so the score is query-relative and NOT a raw cosine: keep it out of
        cosine-calibrated paths (the session-doc merge / threshold filter).

        ``expr`` is an optional server-side metadata pre-filter applied to both legs.
        """
        if k <= 0:
            raise ValueError("k must be positive")
        pool = max(k, _HYBRID_CANDIDATES)
        dense = AnnSearchRequest(
            data=[query],
            anns_field="embedding",
            param=self._search_params(pool),
            limit=pool,
            expr=expr,
        )
        sparse = AnnSearchRequest(
            data=[query_text],
            anns_field=_SPARSE_FIELD,
            param={"metric_type": "BM25"},
            limit=pool,
            expr=expr,
        )
        fused = self._client.hybrid_search(
            collection_name=self._collection,
            reqs=[dense, sparse],
            ranker=WeightedRanker(alpha, beta, norm_score=True),
            limit=k,
            output_fields=list(_OUTPUT_FIELDS),
        )
        if not fused or not fused[0]:
            return []
        return [
            Hit(
                id=hit["id"],
                score=float(hit["distance"]),
                metadata=self._merge_metadata(hit["entity"]),
                # Query-relative min-max blend, not a cosine -- tag it so the
                # relevance filter doesn't apply a cosine threshold to it.
                score_kind=ScoreKind.NORMALIZED,
            )
            for hit in fused[0]
        ]

    def lexical_search(self, query_text: str, k: int = 5, *, expr: str = "") -> list[Hit]:
        """Pure BM25 full-text search over ``text``; no embedding needed.

        The lexical-only tool: matches exact terms/acronyms/codes regardless of
        semantics. ``Hit.score`` is the raw BM25 score (unbounded, not a cosine),
        so keep it out of cosine-calibrated paths (the session-doc merge /
        threshold filter). ``expr`` is an optional server-side metadata pre-filter.
        """
        if k <= 0:
            raise ValueError("k must be positive")
        results = self._client.search(
            collection_name=self._collection,
            data=[query_text],
            anns_field=_SPARSE_FIELD,
            limit=k,
            output_fields=list(_OUTPUT_FIELDS),
            search_params={"metric_type": "BM25"},
            filter=expr,
        )
        if not results:
            return []
        return [
            Hit(
                id=hit["id"],
                score=float(hit["distance"]),
                metadata=self._merge_metadata(hit["entity"]),
                # Raw, unbounded BM25 weight -- not a cosine; tag it so the
                # relevance filter doesn't threshold it on the cosine scale.
                score_kind=ScoreKind.BM25,
            )
            for hit in results[0]
        ]

    def delete_by_source(self, source: str) -> int:
        """Delete every chunk whose promoted ``source`` column equals ``source``.

        ``source`` is an indexed VARCHAR column (exact equality), so this is a
        cheap server-side filtered delete. The value is escaped so a path
        containing a quote or backslash can't break out of the filter literal.
        """
        result = self._client.delete(
            collection_name=self._collection,
            filter=f'{_SOURCE_FIELD} == "{_escape_literal(source)}"',
        )
        # Milvus returns either a {"delete_count": N} mapping or a list of pks.
        if isinstance(result, dict):
            return int(result.get("delete_count", 0))
        return len(result)

    def fetch_neighbors(self, source: str, indices: list[int]) -> dict[int, dict[str, Any]]:
        """Return the metadata of the chunks at ``indices`` within one ``source``.

        A server-side filtered ``query`` on the indexed ``source`` + ``chunk_index``
        columns (no vector search), keyed by ``chunk_index`` for stitching. Missing
        indices are simply absent from the result.
        """
        if not indices:
            return {}
        index_list = ", ".join(str(int(i)) for i in indices)
        rows = self._client.query(
            collection_name=self._collection,
            filter=(
                f'{_SOURCE_FIELD} == "{_escape_literal(source)}" '
                f"&& {_CHUNK_INDEX_FIELD} in [{index_list}]"
            ),
            output_fields=list(_OUTPUT_FIELDS),
        )
        result: dict[int, dict[str, Any]] = {}
        for row in rows:
            metadata = self._merge_metadata(row)
            index = metadata.get(_CHUNK_INDEX_FIELD)
            if index is not None:
                result[int(index)] = metadata
        return result

    def all_items(self, limit: int = 100) -> list[tuple[str, dict[str, Any]]]:
        items: list[tuple[str, dict[str, Any]]] = []
        iterator = self._client.query_iterator(
            collection_name=self._collection,
            filter="",
            output_fields=["id", *_OUTPUT_FIELDS],
            batch_size=min(_ITER_BATCH, limit),
        )
        try:
            while len(items) < limit:
                batch = iterator.next()
                if not batch:
                    break
                for row in batch:
                    items.append((row["id"], self._merge_metadata(row)))
                    if len(items) >= limit:
                        break
        finally:
            iterator.close()
        return items
