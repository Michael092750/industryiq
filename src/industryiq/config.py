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

    # PDF text extractor used at ingestion: "docling" (default; layout-aware,
    # emits Markdown with correct reading order/headings -> better retrieval on
    # report PDFs; needs the optional 'docling' extra and falls back to pypdf on
    # any failure) or "pypdf" (fast pure-Python text, no fallback). Ingestion is
    # an offline batch, so the slower default is worth the chunk-quality win.
    pdf_parser: str = "docling"
    # Whether Docling runs OCR while parsing PDFs (on by default, so text in
    # scanned pages and chart/figure bitmaps is captured). Set DOCLING_OCR=0 to
    # skip OCR for a faster born-digital-only ingest.
    #
    # When on, RapidOCR's detection step is forced to limit_type=max so a large
    # embedded bitmap is downscaled (to RapidOCR's internal 2000px ceiling) before
    # inference. Its default (limit_type=min) only ever upscales, so a full-size
    # chart bitmap stays huge and OOMs the ONNX detection tensor (std::bad_alloc).
    # 2000 is the floor RapidOCR allows in max mode -- it can't be set lower.
    docling_ocr: bool = True
    # How many PDF pages Docling rasterizes/processes concurrently. Its default
    # (4) renders four page images at once; on pages with large media (foldout
    # charts, big embedded figures) that 4x concurrency can exhaust memory and
    # fail the whole page with std::bad_alloc -- and a failed page drops its text
    # too, not just its OCR. 1 serializes page processing for the lowest memory
    # footprint (ingestion is an offline batch, so the slowdown is acceptable);
    # raise it on a roomy machine to ingest faster.
    docling_page_batch_size: int = 1
    # OCR render resolution, as a multiple of 72 DPI (Docling renders each OCR page
    # region at this x1.5 internally). Docling hardcodes 3 (=216 DPI, x1.5=324 DPI
    # actual) -- high enough that the renders pile up and OOM/SIGSEGV the process on
    # large reports. 2 cuts OCR memory ~2.3x with little quality loss; 1 (108 DPI)
    # is ~9x lighter but weaker on small text. Applied by patching RapidOcrModel,
    # since the scale is not exposed through Docling's OCR options.
    docling_ocr_scale: int = 2

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
    # Drop retrieved chunks shorter than this many characters: bare Markdown
    # headings and fragments embed close to topical queries but answer nothing, so
    # left in they crowd real paragraphs out of the top-k. The retriever
    # over-fetches, then trims to k. 0 disables the filter.
    retrieval_min_chunk_chars: int = 200

    # Scheduled bulk ingestion: a background loop that periodically scans a folder
    # (path + interval set by an admin via /admin/ingest-job) and ingests new/
    # changed files into the shared KB. ``enabled`` is the master kill-switch for
    # the loop itself; ``poll_seconds`` is how often it checks whether a run is due.
    ingest_scheduler_enabled: bool = True
    ingest_scheduler_poll_seconds: int = 60


def _env_bool(name: str, default: bool) -> bool:
    """Parse a boolean env var; treat 0/false/no/off (any case) as False."""
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in ("0", "false", "no", "off", "")


def get_settings() -> Settings:
    """Build settings from the current environment (read fresh each call)."""
    cors = os.getenv("CORS_ORIGINS")
    cors_origins = tuple(o.strip() for o in cors.split(",")) if cors else ("http://localhost:5173",)
    return Settings(
        debug_api_key=os.getenv("DEBUG_API_KEY"),
        admin_api_key=os.getenv("ADMIN_API_KEY"),
        database_url=os.getenv("DATABASE_URL"),
        vector_backend=os.getenv("VECTOR_BACKEND", "pgvector"),
        pdf_parser=os.getenv("PDF_PARSER", "docling"),
        docling_ocr=_env_bool("DOCLING_OCR", True),
        docling_page_batch_size=int(os.getenv("DOCLING_PAGE_BATCH_SIZE", "1")),
        docling_ocr_scale=int(os.getenv("DOCLING_OCR_SCALE", "2")),
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
        retrieval_min_chunk_chars=int(os.getenv("RETRIEVAL_MIN_CHUNK_CHARS", "200")),
        ingest_scheduler_enabled=_env_bool("INGEST_SCHEDULER_ENABLED", True),
        ingest_scheduler_poll_seconds=int(os.getenv("INGEST_SCHEDULER_POLL_SECONDS", "60")),
    )
