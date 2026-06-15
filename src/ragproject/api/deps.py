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
from ragproject.core.chat.adapters.store_pg import PgConversationStore
from ragproject.core.embeddings import Embedder, FakeEmbedder
from ragproject.core.generation import FakeLLM, GenerativeLLM
from ragproject.core.pgvectorstore import PgVectorStore
from ragproject.core.pipeline import RagPipeline
from ragproject.core.retrieval import Retriever
from ragproject.core.vectorstore import InMemoryVectorStore, VectorStore


def _build_ai_providers(settings: Settings) -> tuple[Embedder, GenerativeLLM]:
    """Choose the embedder + LLM: real Bedrock, or offline fakes (default).

    The LLM is returned as a :class:`GenerativeLLM` (generate *and* stream); the
    pipeline and rewriter use the generate half, chat uses the streaming half.
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
        policy=ChatPolicy(
            k=settings.chat_retrieval_k,
            history_limit=settings.chat_history_turns,
        ),
    )
