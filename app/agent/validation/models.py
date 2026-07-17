"""Data models for answer validation.

Validation is a **deterministic gate** that inspects the
``ReflectionResult`` and decides whether the answer meets
minimum quality thresholds.  It does NOT inspect the answer
text or the retrieved context directly — that is the
Reflection Node's job.

Key design choice — deterministic vs LLM
----------------------------------------
Validation is rule-based, not LLM-powered.  This is deliberate:

1. **Speed** — no LLM call; validation runs in microseconds.
2. **Predictability** — same input always produces the same output.
3. **Cost** — zero token usage.
4. **No circular dependency** — using an LLM to re-validate an
   LLM-evaluated answer would add latency and cost for marginal gain.

The Reflection Node already did the heavy semantic lifting.  Validation
just checks whether the scores cross configurable thresholds.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class ValidationSeverity(StrEnum):
    """How severe is the quality gap?

    Used by the Retry Node (future) to decide how to fix the answer.
    """

    NONE = "none"            # everything passed — no issue
    MINOR = "minor"          # small gap (e.g. confidence slightly low)
    MAJOR = "major"          # significant gap (e.g. ungrounded claims)
    CRITICAL = "critical"    # answer is unusable (e.g. irrelevant, empty)


class ValidationCheck(BaseModel):
    """A single atomic check in the validation pipeline.

    Each check has a name, a boolean result, and an explanation so
    the Retry Node knows *what* failed and *why*.
    """

    name: str
    passed: bool
    detail: str = ""


class ValidationResult(BaseModel):
    """Structured decision produced by the Validation Node.

    This is a **gate decision**, not an evaluation.  It answers the
    question: "Is this answer good enough to deliver to the user?"

    Attributes
    ----------
    passed
        ``True`` when all required checks passed.
    reason
        Human-readable summary of the decision.
    retry_required
        ``True`` when the answer should be regenerated.  The Retry Node
        reads this flag to decide whether to loop back.
    severity
        Categorises the gap so the Retry Node knows how aggressively
        to intervene (rewrite? re-retrieve? regenerate?).
    confidence_score
        Copy of the reflection's confidence_score for convenience.
    failed_checks
        List of ``ValidationCheck`` that did NOT pass.  Empty when
        ``passed=True``.
    passed_checks
        List of ``ValidationCheck`` that passed.
    """

    passed: bool = False
    reason: str = ""
    retry_required: bool = False
    severity: ValidationSeverity = Field(default=ValidationSeverity.NONE)

    confidence_score: float = Field(default=0.0, ge=0.0, le=1.0)

    failed_checks: list[ValidationCheck] = Field(default_factory=list)
    passed_checks: list[ValidationCheck] = Field(default_factory=list)

    # ------------------------------------------------------------------
    # factory helpers
    # ------------------------------------------------------------------

    @classmethod
    def error_result(cls, *, error: str = "") -> ValidationResult:
        """Return a pessimistic result when the validator itself fails.

        ``passed=False, retry_required=True`` — the Retry Node should
        attempt regeneration.  This is a fail-safe: when in doubt,
        try again.
        """
        return cls(
            passed=False,
            reason=f"Validator error — {error}" if error else "Validator error — unknown cause",
            retry_required=True,
            severity=ValidationSeverity.MAJOR,
            confidence_score=0.0,
            failed_checks=[
                ValidationCheck(
                    name="validator_internal",
                    passed=False,
                    detail="The validation engine itself raised an exception.",
                )
            ],
        )

    def to_log_dict(self) -> dict[str, Any]:
        """Return a compact dict suitable for structured logging."""
        return {
            "passed": self.passed,
            "retry_required": self.retry_required,
            "severity": self.severity.value,
            "confidence": self.confidence_score,
            "failed_count": len(self.failed_checks),
            "failed_names": [c.name for c in self.failed_checks],
        }
