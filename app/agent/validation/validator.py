"""Answer Validator — deterministic quality gate.

The ``AnswerValidator`` applies configurable thresholds to a
``ReflectionResult`` and produces a ``ValidationResult``.

Design
------
- **Deterministic.**  No LLM calls — pure rule evaluation.
- **Fast.**  Runs in microseconds; zero token cost.
- **Configurable.**  Thresholds come from ``Settings`` so operators
  can tune strictness without touching code.
- **Fail-safe.**  On unexpected errors returns a pessimistic
  ``ValidationResult`` that triggers retry.

Why not LLM-based?
    Validation is a gate, not an evaluation.  The Reflection Node
    already used an LLM for semantic analysis.  Running a second
    LLM here would double latency and cost for negligible gain.
    Threshold checks are perfectly suited to deterministic rules.

Threshold semantics
-------------------
Each threshold is a *minimum acceptable value*.  If the reflection
score is **below** the threshold, the corresponding check fails.

- ``min_confidence_score`` → confidence_score check
- ``require_grounded`` → grounded == fully_grounded check
- ``require_completeness`` → complete == True check
- ``require_relevance`` → relevant == True check
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from app.agent.reflection.models import GroundedStatus, ReflectionResult
from app.agent.validation.models import (
    ValidationCheck,
    ValidationResult,
    ValidationSeverity,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ValidationThresholds:
    """Immutable bag of thresholds consumed by ``AnswerValidator``.

    All values can be injected from ``Settings`` so the validator
    itself is stateless and testable.
    """

    min_confidence_score: float = 0.6
    require_grounded: bool = True
    require_completeness: bool = True
    require_relevance: bool = True


class AnswerValidator:
    """Deterministic gate that inspects a ``ReflectionResult``.

    Usage::

        validator = AnswerValidator(thresholds=ValidationThresholds(...))
        result = validator.validate(reflection)
        if result.passed:
            deliver(answer)
        else:
            schedule_retry(result)
    """

    def __init__(self, *, thresholds: ValidationThresholds | None = None) -> None:
        self._thresholds = thresholds or ValidationThresholds()

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def validate(self, reflection: ReflectionResult) -> ValidationResult:
        """Run all configured checks against *reflection*.

        Returns
        -------
        ValidationResult
            Always returns a result — on internal error returns a
            pessimistic ``ValidationResult.error_result()`` so the
            graph never halts.
        """
        t_start = time.monotonic()

        try:
            logger.info(
                "Validation started — confidence=%.2f",
                reflection.confidence_score,
            )
            checks = self._run_checks(reflection)
            passed = all(c.passed for c in checks)
            failed = [c for c in checks if not c.passed]
            passed_list = [c for c in checks if c.passed]

            severity = self._compute_severity(failed)
            retry_required = not passed
            reason = self._build_reason(passed, failed)

            result = ValidationResult(
                passed=passed,
                reason=reason,
                retry_required=retry_required,
                severity=severity,
                confidence_score=reflection.confidence_score,
                failed_checks=failed,
                passed_checks=passed_list,
            )

            elapsed = (time.monotonic() - t_start) * 1000
            self._log_result(result, elapsed)
            return result

        except Exception as exc:
            elapsed = (time.monotonic() - t_start) * 1000
            logger.exception(
                "Validation engine error — returning pessimistic result  "
                "latency=%.0fms",
                elapsed,
            )
            return ValidationResult.error_result(error=str(exc))

    # ------------------------------------------------------------------
    # check implementations
    # ------------------------------------------------------------------

    def _run_checks(self, r: ReflectionResult) -> list[ValidationCheck]:
        """Evaluate each configured threshold.

        Checks are independent — one failing does not skip the others.
        """
        checks: list[ValidationCheck] = []

        # 1. Confidence score
        threshold = self._thresholds.min_confidence_score
        if r.confidence_score >= threshold:
            checks.append(
                ValidationCheck(
                    name="confidence_score",
                    passed=True,
                    detail=f"{r.confidence_score:.2f} >= {threshold:.2f}",
                )
            )
        else:
            checks.append(
                ValidationCheck(
                    name="confidence_score",
                    passed=False,
                    detail=f"{r.confidence_score:.2f} < {threshold:.2f}",
                )
            )

        # 2. Groundedness
        if self._thresholds.require_grounded:
            is_grounded = r.grounded == GroundedStatus.FULLY_GROUNDED
            checks.append(
                ValidationCheck(
                    name="grounded",
                    passed=is_grounded,
                    detail=(
                        f"status={r.grounded.value} — must be fully_grounded"
                        if not is_grounded
                        else f"status={r.grounded.value}"
                    ),
                )
            )

        # 3. Completeness
        if self._thresholds.require_completeness:
            checks.append(
                ValidationCheck(
                    name="complete",
                    passed=r.complete,
                    detail=(
                        "answer is complete" if r.complete else "answer is incomplete"
                    ),
                )
            )

        # 4. Relevance
        if self._thresholds.require_relevance:
            checks.append(
                ValidationCheck(
                    name="relevant",
                    passed=r.relevant,
                    detail=(
                        "answer is relevant" if r.relevant else "answer is irrelevant"
                    ),
                )
            )

        return checks

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_severity(failed: list[ValidationCheck]) -> ValidationSeverity:
        """Map the set of failed checks to a severity level.

        The mapping is deliberately simple — the Retry Node will use
        this to decide *how* to retry:

        - NONE     → deliver as-is
        - MINOR    → tweak prompt, regenerate
        - MAJOR    → re-retrieve + regenerate
        - CRITICAL → rewrite question + full pipeline re-run
        """
        if not failed:
            return ValidationSeverity.NONE

        failed_names = {c.name for c in failed}

        # Critical: answer is completely off-topic
        if "relevant" in failed_names:
            return ValidationSeverity.CRITICAL

        # Major: hallucination or empty answer
        if "grounded" in failed_names or "complete" in failed_names:
            return ValidationSeverity.MAJOR

        # Minor: just a bit below confidence threshold
        return ValidationSeverity.MINOR

    @staticmethod
    def _build_reason(passed: bool, failed: list[ValidationCheck]) -> str:
        """Build a human-readable summary."""
        if passed:
            return "All validation checks passed."
        names = [c.name for c in failed]
        reasons = [c.detail for c in failed]
        return f"Failed checks: {', '.join(names)}. Details: {'; '.join(reasons)}"

    @staticmethod
    def _log_result(result: ValidationResult, elapsed_ms: float) -> None:
        """Emit structured log for observability."""
        log_dict = result.to_log_dict()
        logger.info(
            "Validation complete — "
            "passed=%s  retry=%s  severity=%s  confidence=%.2f  "
            "failed=%d (%s)  latency=%.0fms",
            log_dict["passed"],
            log_dict["retry_required"],
            log_dict["severity"],
            log_dict["confidence"],
            log_dict["failed_count"],
            ", ".join(log_dict["failed_names"]) if log_dict["failed_names"] else "none",
            elapsed_ms,
        )
