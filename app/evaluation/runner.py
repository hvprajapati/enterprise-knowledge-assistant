"""Evaluation Runner — orchestrates metric evaluation over samples.

The runner is the central entry point.  It takes a registry of
metrics (built by ``build_metric_registry()``) and applies them
to samples or full datasets.

Design
------
- **Fail-safe.**  One failing metric does not stop the run.  Errors
  are captured in ``MetricResult.error``.
- **Measured.**  Every metric and sample is timed individually.
- **Batch-friendly.**  ``evaluate_dataset()`` processes samples
  sequentially; future versions can add parallelism.
"""

from __future__ import annotations

import logging
import time

from app.evaluation.metrics import MetricFn
from app.evaluation.models import (
    EvaluationReport,
    EvaluationSample,
    MetricResult,
    SampleResult,
)

logger = logging.getLogger(__name__)


class EvaluationRunner:
    """Run evaluation metrics against samples and datasets.

    Usage::

        registry = build_metric_registry(...)
        runner = EvaluationRunner(metrics=registry)
        result = runner.evaluate_sample(sample)
        report = runner.generate_report(dataset_name="benchmark_v1")
    """

    def __init__(self, *, metrics: dict[str, MetricFn]) -> None:
        self._metrics = metrics
        self._results: list[SampleResult] = []
        self._errors: list[str] = []

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def evaluate_sample(self, sample: EvaluationSample) -> SampleResult:
        """Run all configured metrics against a single sample.

        Parameters
        ----------
        sample:
            The evaluation sample to assess.

        Returns
        -------
        SampleResult
            Aggregated metrics for this sample.
        """
        t_start = time.monotonic()
        logger.info(
            "Evaluating sample — question=%.60s...  metrics=%d",
            sample.question,
            len(self._metrics),
        )

        metric_results: list[MetricResult] = []

        for name, metric_fn in self._metrics.items():
            logger.debug("  Running metric: %s", name)
            try:
                result = metric_fn(sample)
                metric_results.append(result)
            except Exception as exc:
                logger.exception("  Metric '%s' crashed", name)
                metric_results.append(
                    MetricResult(
                        metric_name=name,
                        score=0.0,
                        passed=False,
                        error=str(exc),
                    )
                )

        passed = all(m.passed for m in metric_results)
        scores = [m.score for m in metric_results if not m.error]
        overall = sum(scores) / len(scores) if scores else 0.0
        elapsed = (time.monotonic() - t_start) * 1000

        sample_result = SampleResult(
            sample=sample,
            metrics=metric_results,
            overall_passed=passed,
            overall_score=round(overall, 3),
            total_latency_ms=round(elapsed, 1),
        )

        self._results.append(sample_result)
        return sample_result

    def evaluate_dataset(
        self,
        samples: list[EvaluationSample],
        *,
        dataset_name: str = "",
    ) -> list[SampleResult]:
        """Evaluate all samples in a dataset sequentially.

        Parameters
        ----------
        samples:
            The dataset samples.
        dataset_name:
            Label for the report.

        Returns
        -------
        list[SampleResult]
            One result per sample.
        """
        logger.info(
            "Evaluating dataset '%s' — %d samples, %d metrics",
            dataset_name,
            len(samples),
            len(self._metrics),
        )
        self._results = []
        self._errors = []

        for i, sample in enumerate(samples):
            try:
                self.evaluate_sample(sample)
            except Exception as exc:
                logger.exception("Sample %d evaluation crashed", i)
                self._errors.append(f"Sample {i}: {exc}")

        passed = sum(1 for r in self._results if r.overall_passed)
        logger.info(
            "Dataset '%s' complete — %d/%d passed, %d errors",
            dataset_name,
            passed,
            len(samples),
            len(self._errors),
        )

        return list(self._results)

    # ------------------------------------------------------------------
    # report
    # ------------------------------------------------------------------

    def generate_report(
        self,
        *,
        dataset_name: str = "",
    ) -> EvaluationReport:
        """Generate an ``EvaluationReport`` from accumulated results.

        Call after ``evaluate_dataset()`` or multiple
        ``evaluate_sample()`` calls.
        """
        if not self._results:
            return EvaluationReport(dataset_name=dataset_name)

        # Per-metric averages
        metric_scores: dict[str, list[float]] = {}
        for r in self._results:
            for m in r.metrics:
                if not m.error:
                    metric_scores.setdefault(m.metric_name, []).append(m.score)

        average_scores = {
            name: round(sum(scores) / len(scores), 3)
            for name, scores in metric_scores.items()
        }

        # Metric summaries
        metric_summaries = [
            {
                "metric_name": name,
                "average_score": avg,
                "count": len(metric_scores[name]),
            }
            for name, avg in average_scores.items()
        ]

        # Failures
        failures = [
            {
                "question": r.sample.question[:120],
                "overall_score": r.overall_score,
                "failed_metrics": [
                    {"name": m.metric_name, "score": m.score, "reasoning": m.reasoning[:200]}
                    for m in r.failed_metrics
                ],
            }
            for r in self._results
            if not r.overall_passed
        ]

        # Recommendations
        recommendations: list[str] = []
        for name, avg in average_scores.items():
            if avg < 0.5:
                recommendations.append(
                    f"Metric '{name}' average is low ({avg:.2f}). "
                    f"Review retrieval and generation pipeline stages."
                )

        total_latency = sum(r.total_latency_ms for r in self._results)

        passed = sum(1 for r in self._results if r.overall_passed)

        report = EvaluationReport(
            dataset_name=dataset_name,
            total_samples=len(self._results),
            passed_samples=passed,
            failed_samples=len(self._results) - passed,
            average_scores=average_scores,
            metric_summaries=metric_summaries,
            failures=failures,
            recommendations=recommendations,
            evaluator_errors=list(self._errors),
            total_latency_ms=round(total_latency, 1),
        )

        logger.info(
            "Report generated — dataset=%s  samples=%d  passed=%d",
            dataset_name,
            report.total_samples,
            report.passed_samples,
        )

        return report
