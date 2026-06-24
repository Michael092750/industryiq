"""FastAPI entry point: assemble middleware and routers.

This is the single application entry point. It creates the app, configures CORS,
and includes the route modules -- it defines no feature routes of its own. Each
group of routes lives in its own router module:

* :mod:`industryiq.api.auth_routes`  -- email register/login (issues tokens).
* :mod:`industryiq.api.chat_routes`  -- multi-round chat (the user surface).
* :mod:`industryiq.api.admin_routes` -- key-gated ingestion into the shared KB.
* :mod:`industryiq.api.debug_routes` -- engineer-only inspection (hidden).
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from industryiq.api.admin_routes import router as admin_router
from industryiq.api.auth_routes import router as auth_router
from industryiq.api.chat_routes import router as chat_router
from industryiq.api.debug_routes import router as debug_router
from industryiq.api.deps import get_ingestion_service
from industryiq.config import get_settings
from industryiq.core.ingestion import IngestScheduler


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Start the background ingestion scheduler on startup; stop it on shutdown.

    The scheduler polls the admin-set job config and runs the bulk ingest on its
    interval. It is built from the cached ingestion service, so it shares the
    app's embedder + vector store. One uvicorn worker = one scheduler (the
    Dockerfile runs a single process); more workers would each schedule.
    """
    settings = get_settings()
    scheduler: IngestScheduler | None = None
    if settings.ingest_scheduler_enabled:
        scheduler = IngestScheduler(
            get_ingestion_service(),
            poll_seconds=settings.ingest_scheduler_poll_seconds,
        )
        scheduler.start()
    try:
        yield
    finally:
        if scheduler is not None:
            await scheduler.stop()


app = FastAPI(title="IndustryIQ", lifespan=lifespan)

# Allow the browser frontend (a different origin) to call the API.
# Explicit origins from config, plus any localhost port for local dev.
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(get_settings().cors_origins),
    allow_origin_regex=r"http://localhost:\d+",
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


app.include_router(auth_router)
app.include_router(admin_router)
app.include_router(chat_router)
app.include_router(debug_router)
