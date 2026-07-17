from __future__ import annotations

import logging
import time

from fastapi import FastAPI, Request

from app.api.exceptions import register_handlers
from app.api.routes.health import router as health_router
from app.api.routes.index import router as index_router
from app.api.routes.jobs import router as jobs_router
from app.api.routes.query import router as query_router
from app.api.routes.stream import router as stream_router
from app.api.routes.upload import router as upload_router
from app.config.settings import settings

# ---------------------------------------------------------------------------
# app factory
# ---------------------------------------------------------------------------

app = FastAPI(
    title=settings.project_name,
    version=settings.version,
    description=(
        "Production-grade Enterprise Knowledge Assistant — "
        "RAG-powered question answering over your document corpus."
    ),
)

# ---------------------------------------------------------------------------
# middleware — request logging
# ---------------------------------------------------------------------------

logger = logging.getLogger("api")


@app.middleware("http")
async def log_requests(request: Request, call_next):  # type: ignore[no-untyped-def]
    started = time.monotonic()
    response = await call_next(request)
    elapsed = (time.monotonic() - started) * 1000
    logger.info(
        "%s %s → %d  (%.0f ms)",
        request.method,
        request.url.path,
        response.status_code,
        elapsed,
    )
    return response


# ---------------------------------------------------------------------------
# exception handlers
# ---------------------------------------------------------------------------

register_handlers(app)

# ---------------------------------------------------------------------------
# routes
# ---------------------------------------------------------------------------


@app.get("/")
async def root() -> dict[str, str]:
    return {"message": f"Welcome to {settings.project_name}"}


app.include_router(health_router)
app.include_router(
    index_router,
    prefix=settings.api_prefix,
)
app.include_router(
    query_router,
    prefix=settings.api_prefix,
)
app.include_router(
    stream_router,
    prefix=settings.api_prefix,
)
app.include_router(
    jobs_router,
    prefix=settings.api_prefix,
)
app.include_router(
    upload_router,
    prefix=settings.api_prefix,
)
