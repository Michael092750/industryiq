"""FastAPI entry point: assemble middleware and routers.

This is the single application entry point. It creates the app, configures CORS,
and includes the route modules -- it defines no feature routes of its own. Each
group of routes lives in its own router module:

* :mod:`industryiq.api.auth_routes`  -- email register/login (issues tokens).
* :mod:`industryiq.api.chat_routes`  -- multi-round chat (the user surface).
* :mod:`industryiq.api.admin_routes` -- key-gated ingestion into the shared KB.
* :mod:`industryiq.api.debug_routes` -- engineer-only inspection (hidden).
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from industryiq.api.admin_routes import router as admin_router
from industryiq.api.auth_routes import router as auth_router
from industryiq.api.chat_routes import router as chat_router
from industryiq.api.debug_routes import router as debug_router
from industryiq.config import get_settings

app = FastAPI(title="IndustryIQ")

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
