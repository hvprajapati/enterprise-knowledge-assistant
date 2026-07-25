from __future__ import annotations

import logging
import time
import uuid

from fastapi import Depends, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from app.api.auth import verify_api_key
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
# CORS — allow configured origins
# ---------------------------------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Override in production via .env
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# middleware — request ID, logging, timing
# ---------------------------------------------------------------------------

logger = logging.getLogger("api")


@app.middleware("http")
async def request_middleware(request: Request, call_next):  # type: ignore[no-untyped-def]
    # Attach a unique request ID for tracing through logs
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    request.state.request_id = request_id

    started = time.monotonic()
    response = await call_next(request)
    elapsed = (time.monotonic() - started) * 1000

    response.headers["X-Request-ID"] = request_id

    logger.info(
        "%s %s → %d  (%.0f ms)  [%s]",
        request.method,
        request.url.path,
        response.status_code,
        elapsed,
        request_id,
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


# Health check is always public
app.include_router(health_router)

# Protected routes with API key auth
_auth = [Depends(verify_api_key)]

app.include_router(
    index_router,
    prefix=settings.api_prefix,
    dependencies=_auth,
)
app.include_router(
    query_router,
    prefix=settings.api_prefix,
    dependencies=_auth,
)
app.include_router(
    stream_router,
    prefix=settings.api_prefix,
    dependencies=_auth,
)
app.include_router(
    jobs_router,
    prefix=settings.api_prefix,
    dependencies=_auth,
)
app.include_router(
    upload_router,
    prefix=settings.api_prefix,
    dependencies=_auth,
)
