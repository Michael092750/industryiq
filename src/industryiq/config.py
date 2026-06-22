"""Application settings, loaded from environment variables.

Kept deliberately tiny for now; more settings (database URL, provider choice)
are added as later phases need them.
"""

import os
from dataclasses import dataclass

from dotenv import load_dotenv

# Load variables from a local .env file into the environment (if present).
# Real environment variables set by the OS/platform always take precedence.
load_dotenv()


@dataclass(frozen=True)
class Settings:
    """Runtime configuration."""

    # Secret required to access debug endpoints. When None, debug endpoints are
    # disabled entirely (they respond 404). Set DEBUG_API_KEY to enable them.
    debug_api_key: str | None = None

    # Secret required to access admin (ingestion) endpoints. When None, they are
    # disabled (404). Set ADMIN_API_KEY to enable. Admins populate the shared
    # knowledge base; end users never call these.
    admin_api_key: str | None = None

    # Postgres connection string. When None, the app falls back to the in-memory
    # vector store (data does not survive restarts).
    database_url: str | None = None

    # Which vector store to use: "pgvector" (Postgres, the default) or "milvus".
    # pgvector is kept for benchmarking; "milvus" routes the live app to Milvus.
    vector_backend: str = "pgvector"
    # Milvus standalone connection (used when vector_backend == "milvus").
    milvus_uri: str = "http://localhost:19530"
    milvus_token: str | None = None
    milvus_collection: str = "chunks"
    # Vector index method built on the Milvus collection (HNSW, IVF_FLAT, FLAT,
    # ...). Explicit (not AUTOINDEX) so benchmark runs name the index they used.
    milvus_index_type: str = "HNSW"

    # AI provider: "fake" (offline default), "anthropic" (local: Anthropic API
    # key + a local CPU embedder), or "bedrock" (real Amazon Bedrock on AWS).
    provider: str = "fake"
    aws_region: str = "us-east-1"
    bedrock_llm_model_id: str = "us.anthropic.claude-sonnet-4-6"
    bedrock_embed_model_id: str = "amazon.titan-embed-text-v2:0"
    # Anthropic direct-API settings (used when provider == "anthropic"). The key
    # is read from ANTHROPIC_API_KEY; when None the SDK cannot authenticate.
    anthropic_api_key: str | None = None
    anthropic_llm_model_id: str = "claude-sonnet-4-6"

    # Browser origins allowed to call the API (CORS). The Vite dev server default.
    cors_origins: tuple[str, ...] = ("http://localhost:5173",)

    # Authentication: secret used to sign JWT access tokens (HS256). A stable
    # default keeps local dev (and the offline test suite) working out of the
    # box, but it is PUBLIC -- anyone with it can forge a token for any account.
    # OVERRIDE IT in every real deployment by setting JWT_SECRET to a long random
    # value. Tokens expire after ``jwt_expiry_minutes`` (default 24h).
    jwt_secret: str = "dev-insecure-secret-change-me-in-production"  # noqa: S105 (documented)
    jwt_algorithm: str = "HS256"
    jwt_expiry_minutes: int = 60 * 24

    # Multi-round chat: how many recent turns to feed into the prompt, and how
    # many chunks to retrieve per turn.
    chat_history_turns: int = 6
    chat_retrieval_k: int = 5
    # Retrieval routing: "always" (always search) or "llm" (let the model decide).
    chat_router: str = "always"
    # What the knowledge base holds; injected into the LLM router prompt so it can
    # judge whether a question is in scope instead of guessing blind.
    chat_kb_description: str = "industry analysis reports"
    # Drop retrieved context whose top score is below this (0.0 = keep all).
    chat_relevance_threshold: float = 0.0


def get_settings() -> Settings:
    """Build settings from the current environment (read fresh each call)."""
    cors = os.getenv("CORS_ORIGINS")
    cors_origins = tuple(o.strip() for o in cors.split(",")) if cors else ("http://localhost:5173",)
    return Settings(
        debug_api_key=os.getenv("DEBUG_API_KEY"),
        admin_api_key=os.getenv("ADMIN_API_KEY"),
        database_url=os.getenv("DATABASE_URL"),
        vector_backend=os.getenv("VECTOR_BACKEND", "pgvector"),
        milvus_uri=os.getenv("MILVUS_URI", "http://localhost:19530"),
        milvus_token=os.getenv("MILVUS_TOKEN"),
        milvus_collection=os.getenv("MILVUS_COLLECTION", "chunks"),
        milvus_index_type=os.getenv("MILVUS_INDEX_TYPE", "HNSW"),
        provider=os.getenv("RAG_PROVIDER", "fake"),
        aws_region=os.getenv("AWS_REGION", "us-east-1"),
        bedrock_llm_model_id=os.getenv("BEDROCK_LLM_MODEL_ID", "us.anthropic.claude-sonnet-4-6"),
        bedrock_embed_model_id=os.getenv("BEDROCK_EMBED_MODEL_ID", "amazon.titan-embed-text-v2:0"),
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
        anthropic_llm_model_id=os.getenv("ANTHROPIC_LLM_MODEL_ID", "claude-sonnet-4-6"),
        cors_origins=cors_origins,
        jwt_secret=os.getenv("JWT_SECRET", "dev-insecure-secret-change-me-in-production"),
        jwt_algorithm=os.getenv("JWT_ALGORITHM", "HS256"),
        jwt_expiry_minutes=int(os.getenv("JWT_EXPIRY_MINUTES", str(60 * 24))),
        chat_history_turns=int(os.getenv("CHAT_HISTORY_TURNS", "6")),
        chat_retrieval_k=int(os.getenv("CHAT_RETRIEVAL_K", "5")),
        chat_router=os.getenv("CHAT_ROUTER", "always"),
        chat_kb_description=os.getenv("CHAT_KB_DESCRIPTION", "industry analysis reports"),
        chat_relevance_threshold=float(os.getenv("CHAT_RELEVANCE_THRESHOLD", "0.0")),
    )
