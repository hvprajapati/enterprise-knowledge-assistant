"""Data models for the Evaluation Framework.

These models describe evaluation inputs (samples), outputs
(metric results), and aggregated reports.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field


class EvaluationSample(BaseModel):
    """A single evaluation case — one question with ground truth.

    Attributes
    ----------
    question:
        The user's question.
    expected_answer:
        The reference / ground-truth answer.
    generated_answer:
        The answer produced by the RAG system.
    retrieved_context:
        Passages retrieved from the vector store.
    metadata:
        Arbitrary extra data (source document, category, etc.).
    """

    question: str
    expected_answer: str = ""
    generated_answer: str = ""
    retrieved_context: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class MetricResult(BaseModel):
    """Result of a single evaluation metric.

    Attributes
    ----------
    metric_name:
        e.g. "faithfulness", "hallucination", "answer_relevancy".
    score:
        Float score (semantics depend on the metric — higher is
        usually better, but callers should check ``passed``).
    passed:
        ``True`` when ``score >= threshold``.
    threshold:
        The minimum score required to pass.
    reasoning:
        Explanation from the evaluator (LLM-based evaluators
        provide a rationale).
    error:
        Non-empty when the metric evaluation itself failed.
    latency_ms:
        Execution time for this metric.
    """

    metric_name: str
    score: float = 0.0
    passed: bool = False
    threshold: float = 0.6
    reasoning: str = ""
    error: str = ""
    latency_ms: float = 0.0


class SampleResult(BaseModel):
    """Aggregation of all metric results for one ``EvaluationSample``.

    Produced by ``EvaluationRunner.evaluate_sample()``.
    """

    sample: EvaluationSample
    metrics: list[MetricResult] = Field(default_factory=list)
    overall_passed: bool = False
    overall_score: float = 0.0
    total_latency_ms: float = 0.0

    @property
    def failed_metrics(self) -> list[MetricResult]:
        return [m for m in self.metrics if not m.passed]


class EvaluationReport(BaseModel):
    """Aggregated report for a full evaluation run.

    Produced by ``EvaluationRunner.generate_report()``.
    """

    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    dataset_name: str = ""
    total_samples: int = 0
    passed_samples: int = 0
    failed_samples: int = 0
    average_scores: dict[str, float] = Field(default_factory=dict)
    metric_summaries: list[dict[str, Any]] = Field(default_factory=list)
    failures: list[dict[str, Any]] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)
    evaluator_errors: list[str] = Field(default_factory=list)
    total_latency_ms: float = 0.0
