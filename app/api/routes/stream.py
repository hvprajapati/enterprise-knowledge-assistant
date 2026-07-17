"""POST /api/v1/query/stream — SSE streaming RAG endpoint.

Tokens are pushed to the client as they are produced by the LLM using
Server-Sent Events (SSE).
"""

from __future__ import annotations

import json
import logging
from typing import Annotated

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from app.api.dependencies import get_query_service
from app.api.schemas import QueryRequest
from app.llm.exceptions import LLMError
from app.query.service import QueryError, QueryService

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Query"])


@router.post(
    "/query/stream",
    summary="Ask a question and receive a streaming answer via SSE",
    description=(
        "Identical pipeline to ``POST /api/v1/query``, but the LLM "
        "response is streamed as Server-Sent Events.  Each event carries "
        "a ``token`` field with a text chunk.  The final event has "
        "``type: \"done\"`` with summary metadata."
    ),
)
async def stream_query(
    body: QueryRequest,
    qs: Annotated[QueryService, Depends(get_query_service)],
) -> StreamingResponse:
    logger.info("Stream query request received — question=%d chars", len(body.question))

    return StreamingResponse(
        _sse_generator(body.question, qs),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # disable nginx buffering
        },
    )


# ---------------------------------------------------------------------------
# internal SSE generator
# ---------------------------------------------------------------------------


def _sse_generator(question: str, qs: QueryService):  # type: ignore[no-untyped-def]
    """Yield SSE-formatted events from the streaming pipeline."""
    try:
        for token in qs.stream_answer(question):
            payload = json.dumps({"token": token}, ensure_ascii=False)
            yield f"data: {payload}\n\n"

        # Successful completion
        done = json.dumps({"type": "done"}, ensure_ascii=False)
        yield f"data: {done}\n\n"

    except GeneratorExit:
        logger.info("SSE stream — client disconnected")

    except LLMError as exc:
        logger.error("SSE stream — LLM error: %s", exc)
        err = json.dumps(
            {"type": "error", "detail": str(exc), "error_type": type(exc).__name__},
            ensure_ascii=False,
        )
        yield f"data: {err}\n\n"

    except QueryError as exc:
        logger.error("SSE stream — query error: %s", exc)
        err = json.dumps(
            {"type": "error", "detail": str(exc), "error_type": "query_error"},
            ensure_ascii=False,
        )
        yield f"data: {err}\n\n"

    except Exception:
        logger.exception("SSE stream — unhandled error")
        err = json.dumps(
            {"type": "error", "detail": "Internal server error", "error_type": "internal_error"},
            ensure_ascii=False,
        )
        yield f"data: {err}\n\n"
