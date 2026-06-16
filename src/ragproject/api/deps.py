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

from ragproject.config import Settings, get_settings
from ragproject.core.chat import (
    AlwaysRetrieveRouter,
    ChatPolicy,
    ChatService,
    ConversationStore,
    InMemoryConversationStore,
    LlmQueryRewriter,
    LlmRouter,
    RetrievalRouter,
    ThresholdFilter,
)
from ragproject.core.chat.adapters.session_documents import SessionDocuments
from ragproject.core.chat.adapters.store_pg import PgConversationStore
from ragproject.core.embeddings import Embedder, FakeEmbedder
from ragproject.core.generation import FakeLLM, GenerativeLLM
from ragproject.core.pgvectorstore import PgVectorStore
from ragproject.core.pipeline import RagPipeline
from ragproject.core.retrieval import Retriever
from ragproject.core.vectorstore import InMemoryVectorStore, VectorStore


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
        from ragproject.core.bedrock import BedrockEmbedder, BedrockLLM

        embedder: Embedder = BedrockEmbedder(
            model_id=settings.bedrock_embed_model_id, region=settings.aws_region
        )
        llm: GenerativeLLM = BedrockLLM(
            model_id=settings.bedrock_llm_model_id, region=settings.aws_region
        )
        return embedder, llm
    if settings.provider == "anthropic":
        from ragproject.core.anthropic_llm import AnthropicLLM
        from ragproject.core.local_embeddings import LocalEmbedder

        return LocalEmbedder(), AnthropicLLM(
            model_id=settings.anthropic_llm_model_id,
            api_key=settings.anthropic_api_key,
        )
    return FakeEmbedder(), FakeLLM()


def _build_vector_store(settings: Settings, dim: int) -> VectorStore:
    """Choose the vector store: persistent Postgres, or in-memory (default)."""
    if settings.database_url:
        return PgVectorStore(settings.database_url, dim=dim)
    return InMemoryVectorStore()


def _build_conversation_store(settings: Settings) -> ConversationStore:
    """Choose the conversation store: persistent Postgres, or in-memory (default)."""
    if settings.database_url:
        return PgConversationStore(settings.database_url)
    return InMemoryConversationStore()


@lru_cache(maxsize=1)
def get_pipeline() -> RagPipeline:
    """Return the process-wide pipeline (built once, then cached)."""
    settings = get_settings()
    embedder, llm = _build_ai_providers(settings)
    store = _build_vector_store(settings, embedder.dim)
    return RagPipeline(Retriever(embedder, store), llm)


@lru_cache(maxsize=1)
def get_session_documents() -> SessionDocuments:
    """Return the process-wide session-document index (built once, then cached).

    Shared between the chat service (which retrieves from it) and the upload route
    (which adds to it) -- they must be the same in-memory instance.
    """
    embedder, _ = _build_ai_providers(get_settings())
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
    return ChatService(
        retriever=Retriever(embedder, vector_store),
        router=router,
        rewriter=LlmQueryRewriter(llm),
        llm=llm,
        store=_build_conversation_store(settings),
        relevance_filter=ThresholdFilter(settings.chat_relevance_threshold),
        session_documents=get_session_documents(),
        policy=ChatPolicy(
            k=settings.chat_retrieval_k,
            history_limit=settings.chat_history_turns,
        ),
    )
