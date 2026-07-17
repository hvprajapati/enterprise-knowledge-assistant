"""Response models for the QueryService."""

from __future__ import annotations

from pydantic import BaseModel, Field


class QueryResponse(BaseModel):
    """Structured result returned by :meth:`QueryService.answer`.

    Attributes
    ----------
    answer:
        The LLM-generated answer text.
    sources:
        Unique source filenames cited in the answer context.
    retrieved_count:
        Number of candidate chunks retrieved from FAISS.
    reranked_count:
        Number of chunks kept after cross-encoder reranking.
    prompt_tokens:
        Input tokens consumed (``None`` when unavailable).
    completion_tokens:
        Output tokens produced (``None`` when unavailable).
    total_tokens:
        Sum of input + output tokens (``None`` when unavailable).
    latency_ms:
        Total end-to-end wall-clock latency in milliseconds.
    """

    answer: str
    sources: list[str] = Field(default_factory=list)
    retrieved_count: int = 0
    reranked_count: int = 0
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    latency_ms: float = 0.0
