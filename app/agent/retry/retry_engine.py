"""Retry Engine — intelligent pipeline recovery.

The ``RetryEngine`` inspects the ``ValidationResult`` to decide
whether (and how) the pipeline should loop back for a better answer.

Design
------
- **Deterministic.**  No LLM — pure rule-based mapping from severity
  to strategy.  Fast (microseconds), predictable, zero token cost.
- **Targeted.**  Never blindly re-runs the whole pipeline.  Picks the
  minimal intervention: regenerate, re-retrieve, or full pipeline.
- **Bounded.**  Enforces ``max_retries`` to prevent infinite loops.
- **Fail-safe.**  On internal errors, returns ``NO_RETRY`` so the
  user always gets *some* answer.

Why not LLM-based?
    The LLM already evaluated the answer (Reflection).  Deciding *where*
    to loop back is a mechanical decision, not a semantic one.  A lookup
    table is faster, cheaper, and more predictable.
"""

from __future__ import annotations

import logging
import time

from app.agent.reflection.models import ReflectionResult
from app.agent.retry.models import RetryDecision
from app.agent.retry.strategies import resolve_retry_strategy
from app.agent.validation.models import ValidationResult

logger = logging.getLogger(__name__)


class RetryEngine:
    """Decide whether to retry, and which pipeline stage to re-enter.

    Usage::

        engine = RetryEngine(max_retries=2)
        decision = engine.decide_retry(
            validation=validation_result,
            reflection=reflection_result,
            retry_count=1,
        )
        # decision.next_node → "generate_node"
    """

    def __init__(self, *, max_retries: int = 2) -> None:
        if max_retries < 0:
            raise ValueError(f"max_retries must be >= 0, got {max_retries}")
        self._max_retries = max_retries

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def decide_retry(
        self,
        validation: ValidationResult,
        reflection: ReflectionResult,
        retry_count: int,
    ) -> RetryDecision:
        """Inspect quality results and return a retry decision.

        Parameters
        ----------
        validation:
            The validation gate result (passed, severity, failed checks).
        reflection:
            The reflection evaluation (for context — the engine may log
            hints from ``reflection.recommendations`` in the future).
        retry_count:
            How many retries have already been attempted.  When this
            reaches ``max_retries`` the engine forces ``NO_RETRY``.

        Returns
        -------
        RetryDecision
            Always returns a decision.  On internal error returns
            ``RetryDecision.error_fallback()``.
        """
        t_start = time.monotonic()
        logger.info(
            "Retry engine — severity=%s  passed=%s  retries=%d/%d",
            validation.severity.value,
            validation.passed,
            retry_count,
            self._max_retries,
        )

        try:
            decision = resolve_retry_strategy(
                severity=validation.severity,
                retry_count=retry_count,
                max_retries=self._max_retries,
            )

            # Log reflection recommendations if we're about to retry
            if decision.should_retry and reflection.recommendations:
                logger.info(
                    "Reflection recommendations: %s",
                    reflection.recommendations,
                )

            elapsed = (time.monotonic() - t_start) * 1000
            self._log_decision(decision, elapsed)
            return decision

        except Exception:
            elapsed = (time.monotonic() - t_start) * 1000
            logger.exception(
                "Retry engine error — falling back to NO_RETRY  latency=%.0fms",
                elapsed,
            )
            return RetryDecision.error_fallback(retry_count=retry_count)

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _log_decision(decision: RetryDecision, elapsed_ms: float) -> None:
        log_dict = decision.to_log_dict()
        logger.info(
            "Retry decision — should_retry=%s  strategy=%s  next=%s  count=%d/%d  latency=%.0fms",
            log_dict["should_retry"],
            log_dict["strategy"],
            log_dict["next_node"],
            log_dict["retry_count"],
            log_dict["max_retries"],
            elapsed_ms,
        )
