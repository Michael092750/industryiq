"""Dependency wiring for the API.

:func:`get_pipeline` builds the application's :class:`RagPipeline`. The vector
store is chosen from configuration:

* ``DATABASE_URL`` set  -> :class:`PgVectorStore` (Postgres, persists across restarts)
* ``DATABASE_URL`` unset -> :class:`InMemoryVectorStore` (ephemeral, no setup)

The embedder and LLM are still the offline fakes; real providers replace them
later behind the same interfaces. Tests override this dependency.
"""

from functools import lru_cache

from ragproject.config import get_settings
from ragproject.core.embeddings import Embedder, FakeEmbedder
from ragproject.core.generation import LLM, FakeLLM
from ragproject.core.pgvectorstore import PgVectorStore
from ragproject.core.pipeline import RagPipeline
from ragproject.core.retrieval import Retriever
from ragproject.core.vectorstore import InMemoryVectorStore, VectorStore


@lru_cache(maxsize=1)
def get_pipeline() -> RagPipeline:
    """Return the process-wide pipeline (built once, then cached)."""
    settings = get_settings()

    # Choose AI providers: real Bedrock, or offline fakes (default).
    embedder: Embedder
    llm: LLM
    if settings.provider == "bedrock":
        from ragproject.core.bedrock import BedrockEmbedder, BedrockLLM

        embedder = BedrockEmbedder(
            model_id=settings.bedrock_embed_model_id, region=settings.aws_region
        )
        llm = BedrockLLM(model_id=settings.bedrock_llm_model_id, region=settings.aws_region)
    else:
        embedder = FakeEmbedder()
        llm = FakeLLM()

    # Choose the vector store: persistent Postgres, or in-memory (default).
    store: VectorStore
    if settings.database_url:
        store = PgVectorStore(settings.database_url, dim=embedder.dim)
    else:
        store = InMemoryVectorStore()

    return RagPipeline(Retriever(embedder, store), llm)
