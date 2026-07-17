"""Data models for answer reflection.

The models in this module describe the *evaluation* of an answer,
not the answer itself.  Reflection is purely diagnostic — it never
modifies the answer or the retrieval pipeline.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class AnswerQuality(StrEnum):
    """Overall quality judgement for a generated answer."""

    EXCELLENT = "excellent"
    GOOD = "good"
    ADEQUATE = "adequate"
    INADEQUATE = "inadequate"
    IRRELEVANT = "irrelevant"


class GroundedStatus(StrEnum):
    """Whether claims in the answer are supported by retrieved context.

    Values
    ------
    FULLY_GROUNDED
        Every material claim is backed by at least one retrieved chunk.
    PARTIALLY_GROUNDED
        Some claims are supported; others appear unsupported.
    UNGROUNDED
        The answer contains claims with no supporting evidence.
    UNKNOWN
        The evaluator could not determine grounding (e.g. context
        was empty or the answer was a "not found" response).
    """

    FULLY_GROUNDED = "fully_grounded"
    PARTIALLY_GROUNDED = "partially_grounded"
    UNGROUNDED = "ungrounded"
    UNKNOWN = "unknown"


class ReflectionResult(BaseModel):
    """Structured evaluation of a generated answer.

    This is produced by the **Reflection Node** after the generate node
    runs.  It does NOT modify the answer — it only attaches a quality
    report that downstream nodes (Validation, Retry) can read.

    Attributes
    ----------
    answer_quality
        Overall quality category.
    grounded
        Whether the answer's claims are supported by retrieved context.
    complete
        ``True`` when the answer fully addresses the question.
    relevant
        ``True`` when the answer stays on-topic for the question.
    confidence_score
        Float in [0.0, 1.0] representing the evaluator's confidence
        that this is a satisfactory answer.
    missing_information
        Topics or facts the answer ought to have included but didn't.
    recommendations
        Concrete suggestions for improving the answer (e.g. "retrieve
        with broader search terms", "ask for clarification").
    reasoning
        Free-text explanation of the evaluation — the evaluator's
        chain-of-thought.
    """

    answer_quality: AnswerQuality = Field(default=AnswerQuality.ADEQUATE)
    grounded: GroundedStatus = Field(default=GroundedStatus.UNKNOWN)
    complete: bool = False
    relevant: bool = True
    confidence_score: float = Field(default=0.5, ge=0.0, le=1.0)

    missing_information: list[str] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)
    reasoning: str = ""

    # ------------------------------------------------------------------
    # factory helpers
    # ------------------------------------------------------------------

    @classmethod
    def default_result(
        cls,
        *,
        question: str = "",
        answer: str = "",
        error: str = "",
    ) -> ReflectionResult:
        """Safe fallback used when the evaluator itself fails.

        Returns a neutral result that does NOT block downstream
        processing — the Validation/Retry nodes can decide how to
        handle the uncertainty.
        """
        reason = (
            f"Reflection engine failed — defaulting to neutral evaluation. "
            f"Error: {error if error else 'unknown'}. "
            f"Question was {len(question)} chars, answer was {len(answer)} chars."
        )
        return cls(
            answer_quality=AnswerQuality.ADEQUATE,
            grounded=GroundedStatus.UNKNOWN,
            complete=False,
            relevant=True,
            confidence_score=0.5,
            missing_information=[],
            recommendations=[
                "Reflection engine unavailable — manual review recommended."
            ],
            reasoning=reason,
        )

    def to_log_dict(self) -> dict[str, Any]:
        """Return a compact dict suitable for structured logging."""
        return {
            "quality": self.answer_quality.value,
            "grounded": self.grounded.value,
            "complete": self.complete,
            "relevant": self.relevant,
            "confidence": self.confidence_score,
            "missing_count": len(self.missing_information),
            "rec_count": len(self.recommendations),
        }
