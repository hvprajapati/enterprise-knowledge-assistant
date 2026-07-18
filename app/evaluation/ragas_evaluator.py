"""RAGAS Evaluator — thin wrapper around RAGAS metrics.

Provides a unified interface for running RAGAS evaluation over
samples or datasets.  The actual metric functions are built by
``metrics._make_ragas_metric()`` and registered via
``build_metric_registry()``.

RAGAS (Retrieval Augmented Generation Assessment) is the leading
open-source framework for evaluating RAG pipelines.  Its core
metrics are:

- **Faithfulness** — are all claims in the answer supported by context?
- **Answer Relevancy** — does the answer address the question?
- **Context Precision** — are relevant passages ranked higher?
- **Context Recall** — were all relevant passages retrieved?

RAGAS uses LLM-as-judge — it calls an LLM to score each metric.
"""

from __future__ import annotations

import logging
from typing import Any

from app.evaluation.metrics import build_metric_registry
from app.evaluation.models import EvaluationSample, SampleResult
from app.evaluation.runner import EvaluationRunner

logger = logging.getLogger(__name__)

# RAGAS metric names
RAGAS_METRICS = [
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
]


class RagasEvaluator:
    """Run RAGAS-specific metrics against evaluation samples.

    Usage::

        evaluator = RagasEvaluator(thresholds={"faithfulness": 0.7, ...})
        result = evaluator.evaluate(sample)
    """

    def __init__(
        self,
        thresholds: dict[str, float] | None = None,
    ) -> None:
        self._thresholds = thresholds or {
            "faithfulness": 0.7,
            "answer_relevancy": 0.6,
            "context_precision": 0.6,
            "context_recall": 0.6,
        }
        ragas_metrics = {
            k: v for k, v in build_metric_registry(
                ragas_thresholds=self._thresholds,
            ).items()
            if k in RAGAS_METRICS
        }
        self._runner = EvaluationRunner(metrics=ragas_metrics)

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def evaluate(self, sample: EvaluationSample) -> SampleResult:
        """Run all RAGAS metrics on a single sample."""
        logger.info("RAGAS evaluation — faithfulness, relevancy, precision, recall")
        return self._runner.evaluate_sample(sample)

    def evaluate_dataset(
        self,
        samples: list[EvaluationSample],
        *,
        dataset_name: str = "",
    ) -> list[SampleResult]:
        """Run RAGAS metrics over a dataset."""
        return self._runner.evaluate_dataset(
            samples, dataset_name=dataset_name
        )

    def generate_report(self, *, dataset_name: str = "") -> Any:
        """Generate a RAGAS evaluation report."""
        return self._runner.generate_report(dataset_name=dataset_name)
