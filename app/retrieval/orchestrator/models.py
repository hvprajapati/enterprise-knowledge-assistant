"""Data models for retrieval planning."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field


class QuestionType(StrEnum):
    SIMPLE = "simple"
    KEYWORD = "keyword"
    METADATA = "metadata"
    AMBIGUOUS = "ambiguous"
    BROAD = "broad"
    COMPARISON = "comparison"
    TROUBLESHOOTING = "troubleshooting"
    FACTUAL = "factual"


class RetrievalPlan(BaseModel):
    """Describes which pipeline stages to execute for a question.

    All boolean fields default to ``False`` — the orchestrator
    explicitly enables the stages it determines are useful.
    """

    question_type: QuestionType = QuestionType.FACTUAL
    rewrite_query: bool = False
    use_self_query: bool = False
    use_multi_query: bool = False
    use_hybrid_search: bool = False
    rerank: bool = True
    use_parent_document: bool = False
    use_context_compression: bool = False
    vector_top_k: int = Field(default=50, ge=1, le=200)
    bm25_top_k: int = Field(default=50, ge=0, le=200)

    @property
    def stages_enabled(self) -> list[str]:
        """Return the names of enabled (True) boolean stages."""
        names: list[str] = []
        for field in (
            "rewrite_query",
            "use_self_query",
            "use_multi_query",
            "use_hybrid_search",
            "rerank",
            "use_parent_document",
            "use_context_compression",
        ):
            if getattr(self, field):
                names.append(field)
        return names

    @classmethod
    def full(cls) -> RetrievalPlan:
        """Return a plan with every stage enabled (fallback default)."""
        return cls(
            question_type=QuestionType.FACTUAL,
            rewrite_query=True,
            use_self_query=True,
            use_multi_query=True,
            use_hybrid_search=True,
            rerank=True,
            use_parent_document=True,
            use_context_compression=True,
        )

    @classmethod
    def minimal(cls) -> RetrievalPlan:
        """Return the cheapest correct plan (dense-only, no extras)."""
        return cls(
            question_type=QuestionType.SIMPLE,
            rerank=True,
        )
