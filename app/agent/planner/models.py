"""Data models for agent planning."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field


class QuestionType(StrEnum):
    FACTUAL = "factual"
    COMPARISON = "comparison"
    SUMMARIZATION = "summarization"
    TROUBLESHOOTING = "troubleshooting"
    METADATA_LOOKUP = "metadata_lookup"
    BROAD_RESEARCH = "broad_research"
    CONVERSATIONAL = "conversational"
    UNKNOWN = "unknown"


class RetrievalStrategy(StrEnum):
    DENSE_ONLY = "dense_only"
    HYBRID = "hybrid"
    MULTI_QUERY = "multi_query"
    METADATA_FILTERED = "metadata_filtered"
    FULL_PIPELINE = "full_pipeline"


class ExecutionPlan(BaseModel):
    """Describes how the agent should execute for a given question.

    Built by the ``Planner`` and stored in ``AgentState`` so downstream
    nodes can read it without re-computing.
    """

    question_type: QuestionType = Field(default=QuestionType.UNKNOWN)
    retrieval_strategy: RetrievalStrategy = Field(default=RetrievalStrategy.DENSE_ONLY)
    requires_rewrite: bool = False
    requires_self_query: bool = False
    requires_multi_query: bool = False
    requires_parent_retrieval: bool = False
    requires_context_compression: bool = False
    expected_tools: list[str] = Field(default_factory=list)
    reasoning: str = ""

    @classmethod
    def default_plan(cls, question: str = "") -> ExecutionPlan:
        """Safe fallback — run the full pipeline."""
        return cls(
            question_type=QuestionType.UNKNOWN,
            retrieval_strategy=RetrievalStrategy.FULL_PIPELINE,
            requires_rewrite=True,
            requires_self_query=True,
            requires_multi_query=True,
            requires_parent_retrieval=True,
            requires_context_compression=True,
            expected_tools=[],
            reasoning=f"Default plan — planner failed for: {question[:80]}",
        )
