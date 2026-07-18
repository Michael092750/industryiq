"""Dependency wiring for the API -- the composition root.

This is the *only* place concrete adapters are chosen and assembled; the core
(`RagPipeline`, `ChatService`) depends solely on abstractions. Selection is
driven by configuration:

* ``RAG_PROVIDER=bedrock`` -> real Bedrock embedder/LLM; otherwise offline fakes.
* ``DATABASE_URL`` set       -> Postgres-backed stores (persist across restarts).
* ``DATABASE_URL`` unset     -> in-memory stores (ephemeral, no setup).

Tests override ``get_pipeline`` / ``get_chat_service`` via FastAPI's
``dependency_overrides``.
"""

from functools import lru_cache
from typing import TYPE_CHECKING, Annotated

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

if TYPE_CHECKING:
    from redis import Redis

from industryiq.config import Settings, get_settings
from industryiq.core.agents import Blackboard, TaskQueue
from industryiq.core.auth import AuthService, User, UserStore
from industryiq.core.auth.adapters.store_memory import InMemoryUserStore
from industryiq.core.auth.adapters.store_pg import PgUserStore
from industryiq.core.chat import (
    AlwaysRetrieveRouter,
    ChatPolicy,
    ChatService,
    ConversationStore,
    InMemoryConversationStore,
    LlmRouter,
    RetrievalRouter,
)
from industryiq.core.chat.adapters.store_pg import PgConversationStore
from industryiq.core.embeddings import Embedder, FakeEmbedder
from industryiq.core.generation import FakeLLM, GenerativeLLM
from industryiq.core.ingestion import IngestionService, IngestStateStore
from industryiq.core.ingestion.adapters.store_memory import InMemoryIngestStateStore
from industryiq.core.ingestion.adapters.store_pg import PgIngestStateStore
from industryiq.core.pgvectorstore import PgVectorStore
from industryiq.core.pipeline import RagPipeline
from industryiq.core.retrieval import (
    LlmQueryRewriter,
    RetrievalService,
    Retriever,
    SessionDocuments,
    SessionDocumentStore,
    ThresholdFilter,
)
from industryiq.core.vectorstore import InMemoryVectorStore, MultiVectorStore, VectorStore


def _build_ai_providers(settings: Settings) -> tuple[Embedder, GenerativeLLM]:
    """Choose the embedder + LLM from ``RAG_PROVIDER``:

    * ``bedrock``   -- real Bedrock (Titan embeddings + Claude), IAM-authed (AWS).
    * ``anthropic`` -- local CPU embeddings (fastembed) + Claude via the Anthropic
      API key. Real generation and retrieval locally, no AWS.
    * anything else -- offline fakes (the default).

    The LLM is returned as a :class:`GenerativeLLM` (generate *and* stream); the
    pipeline and rewriter use the generate half, chat uses the streaming half.
    Provider imports are deferred so each provider's heavy/optional dependencies
    load only when that provider is selected.
    """
    if settings.provider == "bedrock":
        from industryiq.core.bedrock import BedrockEmbedder, BedrockLLM

        embedder: Embedder = BedrockEmbedder(
            model_id=settings.bedrock_embed_model_id, region=settings.aws_region
        )
        llm: GenerativeLLM = BedrockLLM(
            model_id=settings.bedrock_llm_model_id, region=settings.aws_region
        )
        return embedder, llm
    if settings.provider == "anthropic":
        from industryiq.core.anthropic_llm import AnthropicLLM
        from industryiq.core.local_embeddings import LocalEmbedder

        return LocalEmbedder(), AnthropicLLM(
            model_id=settings.anthropic_llm_model_id,
            api_key=settings.anthropic_api_key,
        )
    return FakeEmbedder(), FakeLLM()


def _build_milvus_store(settings: Settings, dim: int) -> VectorStore:
    """Build the Milvus store. The pymilvus import is deferred so it loads only
    when Milvus is actually selected."""
    from industryiq.core.milvusvectorstore import MilvusVectorStore

    return MilvusVectorStore(
        settings.milvus_uri,
        dim=dim,
        collection=settings.milvus_collection,
        token=settings.milvus_token,
        index_type=settings.milvus_index_type,
    )


def _build_vector_store(settings: Settings, dim: int) -> VectorStore:
    """Choose the vector store: Milvus, both (fan-out), persistent Postgres, or
    in-memory (default).

    ``VECTOR_BACKEND=milvus`` routes the live app to Milvus; ``=both`` fans writes
    out to *both* pgvector and Milvus (one ingest loads both for a like-for-like
    benchmark, reads served from pgvector -- see :class:`MultiVectorStore`);
    otherwise the store is Postgres+pgvector when ``DATABASE_URL`` is set, else
    in-memory. pgvector is deliberately kept available so it can be benchmarked
    against Milvus.
    """
    if settings.vector_backend == "milvus":
        return _build_milvus_store(settings, dim)
    if settings.vector_backend == "both":
        if not settings.database_url:
            raise ValueError("VECTOR_BACKEND=both requires DATABASE_URL (for the pgvector leg)")
        # pgvector first => it is the read/primary backend; Milvus is write-only here.
        return MultiVectorStore(
            [PgVectorStore(settings.database_url, dim=dim), _build_milvus_store(settings, dim)]
        )
    if settings.database_url:
        return PgVectorStore(settings.database_url, dim=dim)
    return InMemoryVectorStore()


def _build_conversation_store(settings: Settings) -> ConversationStore:
    """Choose the conversation store: persistent Postgres, or in-memory (default)."""
    if settings.database_url:
        return PgConversationStore(settings.database_url)
    return InMemoryConversationStore()


def _build_ingest_state_store(settings: Settings) -> IngestStateStore:
    """Choose the ingestion state store: persistent Postgres, or in-memory (default).

    Without Postgres the manifest does not survive a restart, so each process
    re-ingests the whole tree -- fine for local dev; the live service uses Postgres
    so the schedule's dedup is durable.
    """
    if settings.database_url:
        return PgIngestStateStore(settings.database_url)
    return InMemoryIngestStateStore()


def _build_user_store(settings: Settings) -> UserStore:
    """Choose the user store: persistent Postgres, or in-memory (default)."""
    if settings.database_url:
        return PgUserStore(settings.database_url)
    return InMemoryUserStore()


@lru_cache(maxsize=1)
def get_redis() -> "Redis | None":
    """Return the process-wide Redis client, or ``None`` when REDIS_URL is unset.

    The single seam through which Redis-backed features reach the server. Cached
    so one client (and its connection pool) is shared process-wide. ``None`` means
    "Redis not configured" -- callers degrade rather than fail, mirroring how the
    stores treat an unset ``DATABASE_URL``. The client is lazy, so building it here
    opens no socket; use :func:`industryiq.core.redis_client.ping` to check
    reachability.
    """
    settings = get_settings()
    if not settings.redis_url:
        return None
    from industryiq.core.redis_client import build_redis_client

    return build_redis_client(settings.redis_url)


@lru_cache(maxsize=1)
def get_blackboard() -> Blackboard:
    """Return the process-wide agent blackboard (shared working memory).

    Redis-backed when REDIS_URL is set -- so agents in different processes share it
    and it survives a restart, with a per-run TTL -- otherwise the in-process
    :class:`InMemoryBlackboard`. Built once, then cached.
    """
    settings = get_settings()
    redis = get_redis()
    if redis is not None:
        from industryiq.core.agents.adapters.blackboard_redis import RedisBlackboard

        return RedisBlackboard(redis, ttl_seconds=settings.agent_blackboard_ttl_seconds)
    from industryiq.core.agents.adapters.blackboard_memory import InMemoryBlackboard

    return InMemoryBlackboard()


@lru_cache(maxsize=1)
def get_task_queue() -> TaskQueue:
    """Return the process-wide agent task queue (supervisor -> workers dispatch).

    Redis Streams when REDIS_URL is set -- so workers in different processes compete
    for the same tasks with at-least-once delivery -- otherwise the in-process
    :class:`InMemoryTaskQueue`. Built once, then cached.
    """
    redis = get_redis()
    if redis is not None:
        from industryiq.core.agents.adapters.queue_redis import RedisTaskQueue

        return RedisTaskQueue(redis)
    from industryiq.core.agents.adapters.queue_memory import InMemoryTaskQueue

    return InMemoryTaskQueue()


@lru_cache(maxsize=1)
def get_auth_service() -> AuthService:
    """Return the process-wide auth service (built once, then cached)."""
    settings = get_settings()
    return AuthService(
        _build_user_store(settings),
        secret=settings.jwt_secret,
        algorithm=settings.jwt_algorithm,
        expiry_minutes=settings.jwt_expiry_minutes,
    )


# Reads "Authorization: Bearer <token>"; auto_error=False so a missing header
# reaches our handler as None (a 401 we control) instead of FastAPI's 403.
_bearer = HTTPBearer(auto_error=False)


def get_current_user(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)],
    auth: Annotated[AuthService, Depends(get_auth_service)],
) -> User:
    """Resolve the bearer token to a user, or reject the request with 401."""
    user = auth.identify(credentials.credentials) if credentials is not None else None
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user


@lru_cache(maxsize=1)
def get_pipeline() -> RagPipeline:
    """Return the process-wide pipeline (built once, then cached)."""
    settings = get_settings()
    embedder, llm = _build_ai_providers(settings)
    store = _build_vector_store(settings, embedder.dim)
    retriever = Retriever(embedder, store, min_chunk_chars=settings.retrieval_min_chunk_chars)
    return RagPipeline(retriever, llm, chunk_min_chars=settings.chunk_min_chars)


@lru_cache(maxsize=1)
def get_ingestion_service() -> IngestionService:
    """Return the process-wide ingestion service (built once, then cached).

    Reuses the cached :func:`get_pipeline`, so the scheduled/admin-triggered
    ingest writes through the same embedder + vector store the live app queries
    with -- no chance of an ingest-vs-query dimension mismatch.
    """
    return IngestionService(get_pipeline(), _build_ingest_state_store(get_settings()))


@lru_cache(maxsize=1)
def get_session_documents() -> SessionDocumentStore:
    """Return the process-wide session-document index (built once, then cached).

    Shared between the chat service (which retrieves from it) and the upload route
    (which adds to it). With Redis configured (REDIS_URL set), that shared index is
    Redis-backed -- so it is visible across processes and survives a restart, with
    a sliding TTL; otherwise it is the in-process :class:`SessionDocuments`, which
    only the one process sees and which is cleared on restart.
    """
    settings = get_settings()
    embedder, _ = _build_ai_providers(settings)
    redis = get_redis()
    if redis is not None:
        from industryiq.core.retrieval.adapters.session_documents_redis import (
            RedisSessionDocumentStore,
        )

        return RedisSessionDocumentStore(
            redis, embedder, ttl_seconds=settings.session_doc_ttl_seconds
        )
    return SessionDocuments(embedder)


@lru_cache(maxsize=1)
def get_chat_service() -> ChatService:
    """Return the process-wide chat service (built once, then cached)."""
    settings = get_settings()
    embedder, llm = _build_ai_providers(settings)
    vector_store = _build_vector_store(settings, embedder.dim)
    router: RetrievalRouter = (
        LlmRouter(llm, settings.chat_kb_description)
        if settings.chat_router == "llm"
        else AlwaysRetrieveRouter()
    )
    # Shared between the retrieval service (which reads it) and ChatService (which
    # clears it on delete) -- one instance, disjoint methods.
    session_documents = get_session_documents()
    retrieval = RetrievalService(
        retriever=Retriever(
            embedder, vector_store, min_chunk_chars=settings.retrieval_min_chunk_chars
        ),
        rewriter=LlmQueryRewriter(llm),
        relevance_filter=ThresholdFilter.from_settings(
            settings.chat_relevance_threshold,
            bm25=settings.chat_bm25_threshold,
            normalized=settings.chat_normalized_threshold,
        ),
        session_documents=session_documents,
    )
    return ChatService(
        retrieval=retrieval,
        router=router,
        llm=llm,
        store=_build_conversation_store(settings),
        session_documents=session_documents,
        policy=ChatPolicy(
            k=settings.chat_retrieval_k,
            history_limit=settings.chat_history_turns,
        ),
    )
