"""Metric definitions and registry for the evaluation framework.

Each metric is a callable that takes an ``EvaluationSample`` and
returns a ``MetricResult``.  Metrics are registered in a central
registry so the runner can discover and invoke them.

Design
------
- **Protocol-based.**  Metrics are plain functions, not classes.
  Signature: ``(sample: EvaluationSample) -> MetricResult``.
- **Lazy imports.**  RAGAS and DeepEval metrics are imported only
  when the metric is actually invoked — the app never crashes at
  import time because a library is missing.
- **Thresholds from settings.**  Each metric has a configurable
  pass/fail threshold.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import Any

from app.evaluation.models import EvaluationSample, MetricResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# metric type
# ---------------------------------------------------------------------------

MetricFn = Callable[[EvaluationSample], MetricResult]

# ---------------------------------------------------------------------------
# threshold helpers
# ---------------------------------------------------------------------------


def _get_threshold(metric_name: str, thresholds: dict[str, float]) -> float:
    """Return the configured threshold for *metric_name*."""
    return thresholds.get(metric_name, 0.6)


# ---------------------------------------------------------------------------
# fallback metric (when a library is unavailable)
# ---------------------------------------------------------------------------


def _fallback_result(
    sample: EvaluationSample,
    metric_name: str,
    error: str,
) -> MetricResult:
    """Return a neutral result when the evaluator can't be loaded."""
    logger.warning("Metric '%s' unavailable: %s", metric_name, error)
    return MetricResult(
        metric_name=metric_name,
        score=0.5,
        passed=False,
        threshold=0.6,
        reasoning=f"Metric unavailable: {error}",
        error=error,
    )


# ---------------------------------------------------------------------------
# RAGAS metric factory
# ---------------------------------------------------------------------------


def _make_ragas_metric(
    metric_name: str,
    thresholds: dict[str, float],
    module_attr: str,
    column_map: dict[str, str],
) -> MetricFn:
    """Create a metric function backed by RAGAS, with graceful fallback."""

    def evaluate(sample: EvaluationSample) -> MetricResult:
        t_start = time.monotonic()
        threshold = _get_threshold(metric_name, thresholds)

        try:
            from ragas import SingleTurnSample
            from ragas import evaluate as ragas_evaluate
            from ragas.metrics import (
                answer_relevancy,
                context_precision,
                context_recall,
                faithfulness,
            )

            metric_map = {
                "faithfulness": faithfulness,
                "answer_relevancy": answer_relevancy,
                "context_precision": context_precision,
                "context_recall": context_recall,
            }
            metric = metric_map.get(metric_name)
            if metric is None:
                raise ValueError(f"Unknown RAGAS metric: {metric_name}")

            ragas_sample = SingleTurnSample(
                user_input=sample.question,
                response=sample.generated_answer,
                retrieved_contexts=sample.retrieved_context,
                reference=sample.expected_answer,
            )

            result = ragas_evaluate(
                metrics=[metric],
                dataset=[ragas_sample],
                show_progress=False,
            )

            score = float(getattr(result, metric_name, 0.0) or 0.0)
            elapsed = (time.monotonic() - t_start) * 1000

            return MetricResult(
                metric_name=metric_name,
                score=min(max(score, 0.0), 1.0),
                passed=score >= threshold,
                threshold=threshold,
                reasoning=f"RAGAS {metric_name} score: {score:.3f}",
                latency_ms=elapsed,
            )

        except Exception as exc:
            elapsed = (time.monotonic() - t_start) * 1000
            logger.exception("RAGAS metric '%s' failed", metric_name)
            return MetricResult(
                metric_name=metric_name,
                score=0.0,
                passed=False,
                threshold=threshold,
                reasoning=f"RAGAS evaluation failed: {exc}",
                error=str(exc),
                latency_ms=elapsed,
            )

    return evaluate


# ---------------------------------------------------------------------------
# DeepEval metric factory
# ---------------------------------------------------------------------------


def _make_deepeval_metric(
    metric_name: str,
    thresholds: dict[str, float],
) -> MetricFn:
    """Create a metric function backed by DeepEval, with graceful fallback."""

    def evaluate(sample: EvaluationSample) -> MetricResult:
        t_start = time.monotonic()
        threshold = _get_threshold(metric_name, thresholds)

        try:
            from deepeval.metrics import (
                AnswerRelevancyMetric,
                FaithfulnessMetric,
                GEval,
                HallucinationMetric,
            )
            from deepeval.test_case import LLMTestCase

            test_case = LLMTestCase(
                input=sample.question,
                actual_output=sample.generated_answer,
                expected_output=sample.expected_answer,
                retrieval_context=list(sample.retrieved_context),
            )

            metric_map: dict[str, Any] = {
                "hallucination": HallucinationMetric(threshold=threshold),
                "faithfulness_de": FaithfulnessMetric(threshold=threshold),
                "answer_relevancy_de": AnswerRelevancyMetric(threshold=threshold),
                "bias": GEval(
                    name="Bias",
                    criteria="Determine if the output contains any bias.",
                    threshold=threshold,
                ),
                "toxicity": GEval(
                    name="Toxicity",
                    criteria="Determine if the output contains toxic content.",
                    threshold=threshold,
                ),
            }

            metric = metric_map.get(metric_name)
            if metric is None:
                raise ValueError(f"Unknown DeepEval metric: {metric_name}")

            metric.measure(test_case)
            score = float(metric.score or 0.0)
            reason = str(metric.reason or "")
            elapsed = (time.monotonic() - t_start) * 1000

            return MetricResult(
                metric_name=metric_name,
                score=min(max(score, 0.0), 1.0),
                passed=score >= threshold,
                threshold=threshold,
                reasoning=reason,
                latency_ms=elapsed,
            )

        except Exception as exc:
            elapsed = (time.monotonic() - t_start) * 1000
            logger.exception("DeepEval metric '%s' failed", metric_name)
            return MetricResult(
                metric_name=metric_name,
                score=0.0,
                passed=False,
                threshold=threshold,
                reasoning=f"DeepEval failed: {exc}",
                error=str(exc),
                latency_ms=elapsed,
            )

    return evaluate


# ---------------------------------------------------------------------------
# simple heuristic metrics (no library required)
# ---------------------------------------------------------------------------


def _context_relevance_heuristic(sample: EvaluationSample, threshold: float) -> MetricResult:
    """Simple overlap-based context relevance check."""
    t_start = time.monotonic()
    question_words = set(sample.question.lower().split())
    if not sample.retrieved_context or not question_words:
        score = 0.0
    else:
        hits = sum(
            1 for ctx in sample.retrieved_context
            if any(w in ctx.lower() for w in question_words)
        )
        score = min(hits / max(len(sample.retrieved_context), 1), 1.0)
    elapsed = (time.monotonic() - t_start) * 1000
    return MetricResult(
        metric_name="context_relevance_heuristic",
        score=score,
        passed=score >= threshold,
        threshold=threshold,
        reasoning=f"Keyword overlap: {score:.2f}",
        latency_ms=elapsed,
    )


# ---------------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------------


def build_metric_registry(
    ragas_thresholds: dict[str, float] | None = None,
    deepeval_thresholds: dict[str, float] | None = None,
) -> dict[str, MetricFn]:
    """Return a registry of name → metric function.

    Parameters
    ----------
    ragas_thresholds:
        e.g. ``{"faithfulness": 0.7, "answer_relevancy": 0.6, ...}``
    deepeval_thresholds:
        e.g. ``{"hallucination": 0.7, "bias": 0.8, ...}``

    Returns
    -------
    dict[str, MetricFn]
        Always returns a dict — unavailable metrics use fallback.
    """
    r_t = ragas_thresholds or {}
    d_t = deepeval_thresholds or {}

    registry: dict[str, MetricFn] = {}

    # RAGAS metrics
    ragas_metrics = [
        "faithfulness", "answer_relevancy",
        "context_precision", "context_recall",
    ]
    for name in ragas_metrics:
        registry[name] = _make_ragas_metric(name, r_t, name, {})

    # DeepEval metrics
    deepeval_names = [
        "hallucination", "faithfulness_de",
        "answer_relevancy_de", "bias", "toxicity",
    ]
    for name in deepeval_names:
        registry[name] = _make_deepeval_metric(name, d_t)

    # Heuristic fallback (always available)
    registry["context_relevance_heuristic"] = (
        lambda s, t=0.5: _context_relevance_heuristic(s, t)  # type: ignore[misc]
    )

    return registry
