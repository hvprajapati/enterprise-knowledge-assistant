"""Centralised exception handlers.

Maps application-layer and provider exceptions to appropriate HTTP
status codes and a uniform ``ErrorResponse`` JSON body.
"""

from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.api.schemas import ErrorResponse
from app.llm.exceptions import (
    LLMAuthenticationError,
    LLMError,
    LLMNetworkError,
    LLMProviderError,
    LLMRateLimitError,
    LLMTimeoutError,
)
from app.query.service import QueryError

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# registration
# ---------------------------------------------------------------------------


def register_handlers(app: FastAPI) -> None:
    """Attach all exception handlers to the FastAPI *app*.

    Order matters — put more-specific exception types **before**
    their base types so Starlette matches correctly.
    """
    app.add_exception_handler(QueryError, _handle_query_error)  # type: ignore[arg-type]
    app.add_exception_handler(LLMAuthenticationError, _handle_llm_auth_error)  # type: ignore[arg-type]
    app.add_exception_handler(LLMRateLimitError, _handle_llm_rate_limit)  # type: ignore[arg-type]
    app.add_exception_handler(LLMTimeoutError, _handle_llm_timeout)  # type: ignore[arg-type]
    app.add_exception_handler(LLMNetworkError, _handle_llm_network)  # type: ignore[arg-type]
    app.add_exception_handler(LLMProviderError, _handle_llm_provider)  # type: ignore[arg-type]
    app.add_exception_handler(LLMError, _handle_llm_generic)  # type: ignore[arg-type]
    app.add_exception_handler(ValueError, _handle_value_error)  # type: ignore[arg-type]
    app.add_exception_handler(FileNotFoundError, _handle_file_not_found)  # type: ignore[arg-type]
    app.add_exception_handler(Exception, _handle_generic)


# ---------------------------------------------------------------------------
# handlers
# ---------------------------------------------------------------------------


async def _handle_query_error(request: Request, exc: QueryError) -> JSONResponse:
    logger.warning("Query error: %s", exc)
    return _json(422, detail=str(exc), error_type="query_error")


async def _handle_llm_auth_error(request: Request, exc: LLMAuthenticationError) -> JSONResponse:
    logger.error("LLM authentication failed: %s", exc)
    return _json(401, detail=str(exc), error_type="authentication_error")


async def _handle_llm_rate_limit(request: Request, exc: LLMRateLimitError) -> JSONResponse:
    logger.warning("LLM rate limited: %s", exc)
    return _json(429, detail=str(exc), error_type="rate_limit_error")


async def _handle_llm_timeout(request: Request, exc: LLMTimeoutError) -> JSONResponse:
    logger.error("LLM timeout: %s", exc)
    return _json(504, detail=str(exc), error_type="timeout_error")


async def _handle_llm_network(request: Request, exc: LLMNetworkError) -> JSONResponse:
    logger.error("LLM network error: %s", exc)
    return _json(502, detail=str(exc), error_type="network_error")


async def _handle_llm_provider(request: Request, exc: LLMProviderError) -> JSONResponse:
    logger.error("LLM provider error: %s", exc)
    return _json(502, detail=str(exc), error_type="provider_error")


async def _handle_llm_generic(request: Request, exc: LLMError) -> JSONResponse:
    logger.error("LLM error: %s", exc)
    return _json(500, detail=str(exc), error_type="llm_error")


async def _handle_value_error(request: Request, exc: ValueError) -> JSONResponse:
    logger.warning("Bad request: %s", exc)
    return _json(400, detail=str(exc), error_type="validation_error")


async def _handle_file_not_found(request: Request, exc: FileNotFoundError) -> JSONResponse:
    logger.warning("Resource not found: %s", exc)
    return _json(404, detail=str(exc), error_type="not_found")


async def _handle_generic(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled exception: %s", exc)
    return _json(500, detail="Internal server error", error_type="internal_error")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _json(status: int, *, detail: str, error_type: str) -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content=ErrorResponse(detail=detail, error_type=error_type).model_dump(),
    )
