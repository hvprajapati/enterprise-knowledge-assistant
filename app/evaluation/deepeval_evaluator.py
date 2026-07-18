"""DeepEval Evaluator — thin wrapper around DeepEval metrics.

Provides a unified interface for running DeepEval evaluation over
samples or datasets.

DeepEval is a comprehensive LLM evaluation framework.  Its key
metrics include:

- **Hallucination** — does the answer contain made-up information?
- **Answer Correctness** — how close is the answer to the expected output?
- **Contextual Relevancy** — is the retrieved context useful?
- **Bias** — does the output contain demographic or ideological bias?
- **Toxicity** — does the output contain toxic or harmful content?

DeepEval uses LLM-as-judge with GEval for custom criteria.
"""

from __future__ import annotations

import logging
from typing import Any

from app.evaluation.metrics import build_metric_registry
from app.evaluation.models import EvaluationSample, SampleResult
from app.evaluation.runner import EvaluationRunner

logger = logging.getLogger(__name__)

# DeepEval metric names
DEEPEVAL_METRICS = [
    "hallucination",
    "faithfulness_de",
    "answer_relevancy_de",
    "bias",
    "toxicity",
]


class DeepEvalEvaluator:
    """Run DeepEval metrics against evaluation samples.

    Usage::

        evaluator = DeepEvalEvaluator(thresholds={"hallucination": 0.7, ...})
        result = evaluator.evaluate(sample)
    """

    def __init__(
        self,
        thresholds: dict[str, float] | None = None,
    ) -> None:
        self._thresholds = thresholds or {
            "hallucination": 0.7,
            "faithfulness_de": 0.7,
            "answer_relevancy_de": 0.6,
            "bias": 0.8,
            "toxicity": 0.9,
        }
        de_metrics = {
            k: v for k, v in build_metric_registry(
                deepeval_thresholds=self._thresholds,
            ).items()
            if k in DEEPEVAL_METRICS
        }
        self._runner = EvaluationRunner(metrics=de_metrics)

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def evaluate(self, sample: EvaluationSample) -> SampleResult:
        """Run all DeepEval metrics on a single sample."""
        logger.info("DeepEval evaluation — hallucination, correctness, bias, toxicity")
        return self._runner.evaluate_sample(sample)

    def evaluate_dataset(
        self,
        samples: list[EvaluationSample],
        *,
        dataset_name: str = "",
    ) -> list[SampleResult]:
        """Run DeepEval metrics over a dataset."""
        return self._runner.evaluate_dataset(
            samples, dataset_name=dataset_name
        )

    def generate_report(self, *, dataset_name: str = "") -> Any:
        """Generate a DeepEval evaluation report."""
        return self._runner.generate_report(dataset_name=dataset_name)
