"""Retry strategy mapping — severity → pipeline re-entry point.

This module implements the core decision logic: given a
``ValidationSeverity`` and the current retry count, what should
we do?

The mapping is intentionally simple.  Each severity maps to the
**minimal** intervention that could fix the problem:

    NONE     → NO_RETRY         (deliver answer)
    MINOR    → GENERATE_AGAIN   (re-prompt the LLM)
    MAJOR    → RETRIEVE_AGAIN   (get better context, then re-generate)
    CRITICAL → FULL_PIPELINE    (re-plan, re-write, re-retrieve, re-generate)

Why targeted retries?
    - MINOR issues (confidence just below threshold) usually mean the
      LLM output was slightly weak — regenerating with the same context
      often produces a better answer.
    - MAJOR issues (ungrounded claims) mean the context didn't support
      the answer — re-retrieving with broader search is appropriate.
    - CRITICAL issues (irrelevant answer) mean the whole pipeline went
      wrong — start fresh from the planner.

Why not always FULL_PIPELINE?
    Latency and cost.  Re-running the full pipeline when only the
    generation was weak wastes ~500ms–2s and doubles token usage.
    Targeted retries fix the broken stage and only the broken stage.
"""

from __future__ import annotations

from app.agent.retry.models import RetryDecision, RetryStrategy
from app.agent.validation.models import ValidationSeverity

# ---------------------------------------------------------------------------
# severity → (should_retry, strategy, reason_template)
# ---------------------------------------------------------------------------

_STRATEGY_MAP: dict[
    ValidationSeverity,
    tuple[bool, RetryStrategy, str],
] = {
    ValidationSeverity.NONE: (
        False,
        RetryStrategy.NO_RETRY,
        "Answer passed all validation checks.",
    ),
    ValidationSeverity.MINOR: (
        True,
        RetryStrategy.GENERATE_AGAIN,
        "Minor quality gap (low confidence) — regenerating answer with same context.",
    ),
    ValidationSeverity.MAJOR: (
        True,
        RetryStrategy.RETRIEVE_AGAIN,
        "Major quality gap (ungrounded or incomplete) — re-retrieving with broader search.",
    ),
    ValidationSeverity.CRITICAL: (
        True,
        RetryStrategy.FULL_PIPELINE,
        "Critical quality gap (irrelevant or empty) — re-running full pipeline.",
    ),
}

# Node name for each strategy (used in conditional routing)
STRATEGY_NODE_MAP: dict[RetryStrategy, str] = {
    RetryStrategy.NO_RETRY: "__end__",
    RetryStrategy.RETRIEVE_AGAIN: "retrieve_node",
    RetryStrategy.GENERATE_AGAIN: "generate_node",
    RetryStrategy.FULL_PIPELINE: "planner_node",
}


def resolve_retry_strategy(
    severity: ValidationSeverity,
    *,
    retry_count: int,
    max_retries: int,
) -> RetryDecision:
    """Map a ``ValidationSeverity`` to a ``RetryDecision``.

    Parameters
    ----------
    severity:
        The severity from ``ValidationResult``.
    retry_count:
        How many retries have already been attempted.
    max_retries:
        The configured retry limit from settings.

    Returns
    -------
    RetryDecision
        Always returns a decision — if the severity is unknown or the
        limit is reached, returns ``NO_RETRY``.
    """
    # Guard: retry limit
    if retry_count >= max_retries:
        return RetryDecision.retry_limit_reached(
            retry_count=retry_count,
            max_retries=max_retries,
        )

    # Look up the strategy
    entry = _STRATEGY_MAP.get(severity)
    if entry is None:
        # Unknown severity — be safe, don't retry
        return RetryDecision.no_retry(
            reason=(f"Unknown validation severity '{severity.value}' — delivering answer as-is."),
            retry_count=retry_count,
            max_retries=max_retries,
        )

    should_retry, strategy, reason_template = entry
    next_node = STRATEGY_NODE_MAP[strategy]

    return RetryDecision(
        should_retry=should_retry,
        strategy=strategy,
        reason=reason_template,
        retry_count=retry_count,
        max_retries=max_retries,
        next_node=next_node,
    )
