"""POST /api/v1/query — RAG question-answering endpoint."""

from __future__ import annotations

import logging
from typing import Annotated

from fastapi import APIRouter, Depends

from app.api.dependencies import get_query_service
from app.api.schemas import QueryRequest, QueryResponse
from app.query.service import QueryService

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Query"])


@router.post(
    "/query",
    response_model=QueryResponse,
    summary="Ask a question against the indexed knowledge base",
    description=(
        "Runs the full RAG pipeline: embed the question, retrieve "
        "candidate passages from FAISS, rerank them with a cross-encoder, "
        "build a grounded prompt, and generate an answer via the "
        "configured LLM provider."
    ),
)
async def ask_question(
    body: QueryRequest,
    qs: Annotated[QueryService, Depends(get_query_service)],
) -> QueryResponse:
    logger.info("Query request received — question=%d chars", len(body.question))
    return qs.answer(body.question)
