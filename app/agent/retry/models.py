"""Data models for intelligent retry decisions.

The Retry Node consumes ``ValidationResult`` and ``ReflectionResult``
and produces a ``RetryDecision`` that routes execution back to the
most appropriate pipeline stage.

Why not just always FULL_PIPELINE?
    Blindly re-running everything wastes latency and tokens.  If the
    retrieval was fine but the generation was poor, re-retrieving is
    pointless.  Targeted retries fix the specific broken stage.

Why deterministic?
    Same reasons as Validation — speed, predictability, zero token
    cost.  The LLM already evaluated the answer; we just need to
    decide *where* to loop back.  That's a simple mapping.

Retry limits
    ``max_retries`` from Settings caps the number of loops.  Without
    this, a consistently bad retrieval could loop forever.  Once the
    limit is hit, the engine returns ``NO_RETRY`` regardless of score.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class RetryStrategy(StrEnum):
    """What part of the pipeline should be re-executed?

    Values map to LangGraph node names for direct conditional routing.
    """

    NO_RETRY = "no_retry"  # → END
    RETRIEVE_AGAIN = "retrieve_again"  # → retrieve_node
    GENERATE_AGAIN = "generate_again"  # → generate_node
    FULL_PIPELINE = "full_pipeline"  # → planner_node


class RetryDecision(BaseModel):
    """Decision produced by the Retry Engine.

    Attributes
    ----------
    should_retry
        ``True`` when the engine recommends re-executing part of the
        pipeline.  ``False`` means deliver the answer as-is (or give up).
    strategy
        Which pipeline stage to re-enter.
    reason
        Human-readable explanation (e.g. "Low confidence — regenerate").
    retry_count
        How many retries have been performed so far (from state).
    max_retries
        Configured limit — when ``retry_count >= max_retries`` the
        engine forces ``NO_RETRY``.
    next_node
        The LangGraph node name to route to.  One of:
        ``"planner_node"``, ``"retrieve_node"``, ``"generate_node"``,
        or ``"__end__"``.
    """

    should_retry: bool = False
    strategy: RetryStrategy = Field(default=RetryStrategy.NO_RETRY)
    reason: str = ""
    retry_count: int = 0
    max_retries: int = 2
    next_node: str = "__end__"

    # ------------------------------------------------------------------
    # factory helpers
    # ------------------------------------------------------------------

    @classmethod
    def no_retry(
        cls,
        *,
        reason: str = "Answer acceptable — no retry needed.",
        retry_count: int = 0,
        max_retries: int = 2,
    ) -> RetryDecision:
        """Return a decision that delivers the current answer."""
        return cls(
            should_retry=False,
            strategy=RetryStrategy.NO_RETRY,
            reason=reason,
            retry_count=retry_count,
            max_retries=max_retries,
            next_node="__end__",
        )

    @classmethod
    def retry_limit_reached(
        cls,
        *,
        retry_count: int = 0,
        max_retries: int = 2,
    ) -> RetryDecision:
        """Return a decision that gives up after hitting the retry cap."""
        return cls(
            should_retry=False,
            strategy=RetryStrategy.NO_RETRY,
            reason=(
                f"Retry limit reached ({retry_count}/{max_retries}). "
                "Delivering best available answer."
            ),
            retry_count=retry_count,
            max_retries=max_retries,
            next_node="__end__",
        )

    @classmethod
    def error_fallback(cls, *, retry_count: int = 0) -> RetryDecision:
        """Return a safe fallback when the retry engine itself fails."""
        return cls(
            should_retry=False,
            strategy=RetryStrategy.NO_RETRY,
            reason="Retry engine error — delivering current answer to avoid infinite loop.",
            retry_count=retry_count,
            max_retries=2,
            next_node="__end__",
        )

    def to_log_dict(self) -> dict[str, Any]:
        """Return a compact dict suitable for structured logging."""
        return {
            "should_retry": self.should_retry,
            "strategy": self.strategy.value,
            "retry_count": self.retry_count,
            "max_retries": self.max_retries,
            "next_node": self.next_node,
            "reason": self.reason[:120],
        }
