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

    # Postgres connection string. When None, the app falls back to the in-memory
    # vector store (data does not survive restarts).
    database_url: str | None = None

    # AI provider: "fake" (offline, default) or "bedrock" (real Amazon Bedrock).
    provider: str = "fake"
    aws_region: str = "us-east-1"
    bedrock_llm_model_id: str = "us.anthropic.claude-sonnet-4-6"
    bedrock_embed_model_id: str = "amazon.titan-embed-text-v2:0"

    # Browser origins allowed to call the API (CORS). The Vite dev server default.
    cors_origins: tuple[str, ...] = ("http://localhost:5173",)


def get_settings() -> Settings:
    """Build settings from the current environment (read fresh each call)."""
    cors = os.getenv("CORS_ORIGINS")
    cors_origins = tuple(o.strip() for o in cors.split(",")) if cors else ("http://localhost:5173",)
    return Settings(
        debug_api_key=os.getenv("DEBUG_API_KEY"),
        database_url=os.getenv("DATABASE_URL"),
        provider=os.getenv("RAG_PROVIDER", "fake"),
        aws_region=os.getenv("AWS_REGION", "us-east-1"),
        bedrock_llm_model_id=os.getenv("BEDROCK_LLM_MODEL_ID", "us.anthropic.claude-sonnet-4-6"),
        bedrock_embed_model_id=os.getenv("BEDROCK_EMBED_MODEL_ID", "amazon.titan-embed-text-v2:0"),
        cors_origins=cors_origins,
    )
