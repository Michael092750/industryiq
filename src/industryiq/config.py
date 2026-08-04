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


# What the knowledge base actually holds -- injected into the turn-router and
# strategy-router prompts so both judge scope and pick metadata filters on the real
# corpus shape rather than a vague blurb. Names the five sector *categories*, the
# publisher *types* (which map to the ``source_type`` facet), example *publishers*
# (which map to the ``publisher`` domain facet), the rough *date* span (the
# ``published_date`` year facet), and the fact-dense content (why exact-figure
# lookups favour lexical search). Keep it aligned with the ingested corpus.
_DEFAULT_KB_DESCRIPTION = (
    "industry and market-research reports across five sectors -- artificial "
    "intelligence, finance and banking, healthcare, agriculture, and semiconductors "
    "-- published mainly 2019-2025 by management consultancies (e.g. McKinsey, BCG, "
    "Bain, Deloitte, EY), industry associations, academic and think-tank groups "
    "(e.g. Stanford HAI, WEF), and government and regulatory bodies (e.g. FDIC, the "
    "Federal Reserve, IMF); the reports are dense with market sizes, forecasts, "
    "investment and adoption figures, and financial statistics"
)


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

    # Redis connection string (e.g. redis://localhost:6379/0). Redis is the hot,
    # shared, ephemeral tier for agent working-memory and cross-agent context.
    # When None, Redis-backed features are disabled -- mirroring the database_url
    # seam: the app runs without it, callers of get_redis() get None.
    redis_url: str | None = None
    # Sliding TTL (seconds) for a conversation's uploaded session documents when
    # they live in Redis (RedisSessionDocumentStore). Refreshed on each upload, so
    # idle sessions self-evict while active ones persist. Default 7 days. Ignored
    # by the in-memory session store (which is cleared on restart anyway).
    session_doc_ttl_seconds: int = 60 * 60 * 24 * 7
    # Sliding TTL (seconds) for an agent run's Redis blackboard namespace, refreshed
    # on each write, so a finished/abandoned run's scratch state self-evicts. Default
    # 1 day. Ignored by the in-memory blackboard.
    agent_blackboard_ttl_seconds: int = 60 * 60 * 24

    # Multi-agent orchestrator (supervisor + workers over the task queue).
    # How long a task may sit claimed-but-unacked before a live worker reclaims it
    # (the crash-recovery window). ``max_attempts`` caps redeliveries before a task
    # is dead-lettered. ``run_timeout_s`` bounds how long the supervisor waits for a
    # run to finish. ``capability_k`` is the retrieval depth of each subtask's mini-RAG.
    agent_reclaim_min_idle_ms: float = 5000.0
    agent_max_attempts: int = 3
    agent_run_timeout_s: float = 30.0
    agent_worker_batch: int = 4
    agent_capability_k: int = 6
    # Max Anthropic server-side web searches per web_search capability call. The
    # web_search tool is registered only when an ANTHROPIC_API_KEY is available.
    agent_web_search_max_uses: int = 5
    # Demo failure injection (leave "off" in production). "crash_once" makes a worker
    # (or the local executor) fail the first attempt of each node, to stage the
    # kill-a-worker beat: Option C reclaims and resumes; Option B loses the run.
    agent_failure_mode: str = "off"
    # Which executor answers a COMPLEX chat turn (one the router flags needs_planning):
    # "local" (in-process LocalExecutor -- low latency, the default) or "distributed"
    # (Supervisor + worker queue -- needs workers + Redis running).
    chat_agent_executor: str = "local"

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
    # Hybrid text recovery (docling parser only; on by default). Docling routes
    # charts/figures/boxed callouts to <!-- image --> and discards the text inside
    # them -- so figure-attached prose (chart footnotes, source notes, average-line
    # labels, timeline entries) is lost, even though the PDF's text layer has it.
    # pypdf reads that layer flat, so after the Docling parse we append, per page,
    # the pypdf lines Docling dropped (deduped against the Docling text and against
    # running headers/footers). This is a lossless completeness net for exactly the
    # numeric facts the figure-VLM pass can't reach; it never touches Docling's
    # clean reading order. Set PDF_HYBRID_RECOVERY=0 for a Docling-only parse.
    pdf_hybrid_recovery: bool = True
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
    # Whether Docling extracts data from charts/figures the layout model detects as
    # pictures. On, each chart is run through a vision model (Granite Vision V4) and
    # emitted as a CSV of its values, so numeric facts that live *inside* a chart
    # (which the plain layout export drops as an <!-- image --> placeholder) become
    # retrievable text. Off by default: it downloads a multi-GB model and runs a
    # vision model per figure, adding significant time + memory (raises the OOM/
    # SIGSEGV risk this pipeline already guards against) -- enable for chart-heavy
    # corpora and ingest through scripts/ingest_resilient.py.
    docling_chart_extraction: bool = False
    # Whether Docling captions each detected picture with a vision-language model
    # (its description enters the document text). Complements chart extraction for
    # non-chart figures (diagrams, photos). Same cost caveat -- a VLM runs per
    # picture. The default model is a small local SmolVLM; off by default.
    docling_picture_description: bool = False

    # Figure understanding at ingest via a vision model (VLM), run as a *separate* pass
    # over docling's finished document -- NOT inside its pipeline. "off" (default) leaves
    # figures as <!-- image --> placeholders. "anthropic" makes one Claude vision call per
    # detected figure and splices the result in place: a data chart/table image becomes a
    # Markdown table of its values (recovering the numeric facts that live inside charts);
    # any other figure becomes a one-line description. This replaces docling's local
    # chart-extraction/picture-description stages (keep both above OFF) with an off-box call
    # that can't crash the parse. See docs/figure-ingestion.md.
    figure_vlm: str = "off"
    figure_vlm_model: str = "claude-sonnet-5"
    # Skip figures whose long edge is under this many pixels (logos, icons, rules): a VLM
    # call on them wastes tokens and yields sub-threshold noise chunks.
    figure_vlm_min_pixels: int = 200
    # Cap the number of figures transcribed per document (0 = no cap). For cost-bounded
    # test runs; leave 0 in production.
    figure_vlm_max_figures: int = 0

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
    # Chunks retrieved per turn. 8 (was 5): the bottleneck analysis found answer
    # chunks landing at rank 6-7 for near-duplicate corpora (e.g. FDIC quarterly
    # reports), just outside a top-5, so a slightly wider window recovers them.
    chat_retrieval_k: int = 8
    # Retrieval routing: "always" (always search) or "llm" (let the model decide).
    chat_router: str = "always"
    # Search-strategy routing (how to search, once we've decided to): "fixed" (always
    # hybrid-RRF, today's behaviour) or "llm" (classify strategy + metadata filter +
    # weights per question). Only "llm" against a Milvus-class store exercises the
    # lexical/weighted/filtered paths; other stores raise on a non-default plan.
    chat_strategy_router: str = "fixed"
    # What the knowledge base holds; injected into the LLM router prompt so it can
    # judge whether a question is in scope instead of guessing blind. See
    # _DEFAULT_KB_DESCRIPTION for the corpus-shaped default.
    chat_kb_description: str = _DEFAULT_KB_DESCRIPTION
    # Drop retrieved context whose top score is below this (0.0 = keep all). This
    # is the *cosine* cutoff -- it only applies to cosine-scored hits (dense/hybrid).
    chat_relevance_threshold: float = 0.0
    # Per-scale cutoffs for the non-cosine strategies (see
    # industryiq.core.vectorstore.ScoreKind). BM25 is an unbounded lexical weight
    # and the weighted blend is a query-relative [0, 1] score, so the cosine cutoff
    # above is meaningless for them. ``None`` = keep all of that kind (the safe
    # default). Starting points to tune on the benchmark: ~1.0 for BM25, ~0.2 for
    # the normalized blend. Only bite once a search strategy actually returns those
    # score kinds.
    chat_bm25_threshold: float | None = None
    chat_normalized_threshold: float | None = None
    # Context (neighbour) expansion: widen each retrieved hit with its adjacent chunks
    # so a fact straddling a chunk boundary is still in the grounding context. Off by
    # default (changes what the generator sees). ``radius`` = neighbours per side;
    # ``max_chunks`` caps the odd window width per hit so k hits can't blow the context.
    chat_context_expansion: bool = False
    chat_context_radius: int = 1
    chat_context_max_chunks: int = 5
    # Drop retrieved chunks shorter than this many characters: bare Markdown
    # headings, short one-line figure descriptions, and other fragments embed close
    # to topical queries but answer nothing, so left in they crowd real paragraphs
    # (and full figure tables) out of the top-k. The retriever over-fetches, then
    # trims to k. 0 disables the filter. This is the query-time *band-aid*; the real
    # fix is ``chunk_min_chars`` below, which merges short chunks at ingest so few
    # remain to filter. 200 (was 400): the bottleneck analysis found a 275-char chunk
    # holding the answer (and short figure-VLM chunks) being trimmed away at 400 --
    # kept low enough to pass terse exact-fact chunks, high enough to drop bare headings.
    retrieval_min_chunk_chars: int = 200
    # Ingest-time chunk coalescing floor: merge adjacent chunks (a small figure table,
    # a one-line caption, a short trailing remainder) until each reaches this many
    # characters, so no orphan short chunk is produced in the first place -- the small
    # table rides *with* its surrounding prose (better embedding + still retrievable)
    # instead of being dropped by the retrieval filter. Never splits a table. 0 disables
    # coalescing. See industryiq.core.chunking.chunk_markdown.
    chunk_min_chars: int = 400

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


def _env_float_opt(name: str) -> float | None:
    """Parse an optional float env var; unset or blank -> ``None`` (no cutoff)."""
    raw = os.getenv(name)
    return float(raw) if raw is not None and raw.strip() != "" else None


def get_settings() -> Settings:
    """Build settings from the current environment (read fresh each call)."""
    cors = os.getenv("CORS_ORIGINS")
    cors_origins = tuple(o.strip() for o in cors.split(",")) if cors else ("http://localhost:5173",)
    return Settings(
        debug_api_key=os.getenv("DEBUG_API_KEY"),
        admin_api_key=os.getenv("ADMIN_API_KEY"),
        database_url=os.getenv("DATABASE_URL"),
        redis_url=os.getenv("REDIS_URL"),
        session_doc_ttl_seconds=int(os.getenv("SESSION_DOC_TTL_SECONDS", str(60 * 60 * 24 * 7))),
        agent_blackboard_ttl_seconds=int(
            os.getenv("AGENT_BLACKBOARD_TTL_SECONDS", str(60 * 60 * 24))
        ),
        agent_reclaim_min_idle_ms=float(os.getenv("AGENT_RECLAIM_MIN_IDLE_MS", "5000")),
        agent_max_attempts=int(os.getenv("AGENT_MAX_ATTEMPTS", "3")),
        agent_run_timeout_s=float(os.getenv("AGENT_RUN_TIMEOUT_S", "30")),
        agent_worker_batch=int(os.getenv("AGENT_WORKER_BATCH", "4")),
        agent_capability_k=int(os.getenv("AGENT_CAPABILITY_K", "6")),
        agent_web_search_max_uses=int(os.getenv("AGENT_WEB_SEARCH_MAX_USES", "5")),
        agent_failure_mode=os.getenv("AGENT_FAILURE_MODE", "off"),
        chat_agent_executor=os.getenv("CHAT_AGENT_EXECUTOR", "local"),
        vector_backend=os.getenv("VECTOR_BACKEND", "pgvector"),
        pdf_parser=os.getenv("PDF_PARSER", "docling"),
        pdf_hybrid_recovery=_env_bool("PDF_HYBRID_RECOVERY", True),
        docling_ocr=_env_bool("DOCLING_OCR", True),
        docling_page_batch_size=int(os.getenv("DOCLING_PAGE_BATCH_SIZE", "1")),
        docling_ocr_scale=int(os.getenv("DOCLING_OCR_SCALE", "2")),
        docling_chart_extraction=_env_bool("DOCLING_CHART_EXTRACTION", False),
        docling_picture_description=_env_bool("DOCLING_PICTURE_DESCRIPTION", False),
        figure_vlm=os.getenv("FIGURE_VLM", "off"),
        figure_vlm_model=os.getenv("FIGURE_VLM_MODEL", "claude-sonnet-5"),
        figure_vlm_min_pixels=int(os.getenv("FIGURE_VLM_MIN_PIXELS", "200")),
        figure_vlm_max_figures=int(os.getenv("FIGURE_VLM_MAX_FIGURES", "0")),
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
        chat_retrieval_k=int(os.getenv("CHAT_RETRIEVAL_K", "8")),
        chat_router=os.getenv("CHAT_ROUTER", "always"),
        chat_strategy_router=os.getenv("CHAT_STRATEGY_ROUTER", "fixed"),
        chat_kb_description=os.getenv("CHAT_KB_DESCRIPTION", _DEFAULT_KB_DESCRIPTION),
        chat_relevance_threshold=float(os.getenv("CHAT_RELEVANCE_THRESHOLD", "0.0")),
        chat_bm25_threshold=_env_float_opt("CHAT_BM25_THRESHOLD"),
        chat_normalized_threshold=_env_float_opt("CHAT_NORMALIZED_THRESHOLD"),
        chat_context_expansion=_env_bool("CHAT_CONTEXT_EXPANSION", False),
        chat_context_radius=int(os.getenv("CHAT_CONTEXT_RADIUS", "1")),
        chat_context_max_chunks=int(os.getenv("CHAT_CONTEXT_MAX_CHUNKS", "5")),
        retrieval_min_chunk_chars=int(os.getenv("RETRIEVAL_MIN_CHUNK_CHARS", "200")),
        chunk_min_chars=int(os.getenv("CHUNK_MIN_CHARS", "400")),
        ingest_scheduler_enabled=_env_bool("INGEST_SCHEDULER_ENABLED", True),
        ingest_scheduler_poll_seconds=int(os.getenv("INGEST_SCHEDULER_POLL_SECONDS", "60")),
    )
