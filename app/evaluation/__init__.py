"""Evaluation Framework — assess RAG pipeline quality.

Components
----------
- ``EvaluationSample`` — one question-answer-context tuple.
- ``MetricResult`` — one metric's score, pass/fail, reasoning.
- ``EvaluationReport`` — aggregated results for a dataset.
- ``EvaluationRunner`` — orchestrates metric evaluation.
- ``RagasEvaluator`` — faithfulness, answer relevancy, context precision/recall.
- ``DeepEvalEvaluator`` — hallucination, bias, toxicity, correctness.
- ``ReportWriter`` — JSON + Markdown report generation.
- ``build_metric_registry`` — factory for metric functions.
"""

from __future__ import annotations

from app.evaluation.dataset import build_samples_from_dicts, load_dataset_from_json
from app.evaluation.deepeval_evaluator import DeepEvalEvaluator
from app.evaluation.metrics import build_metric_registry
from app.evaluation.models import (
    EvaluationReport,
    EvaluationSample,
    MetricResult,
    SampleResult,
)
from app.evaluation.ragas_evaluator import RagasEvaluator
from app.evaluation.report import ReportWriter
from app.evaluation.runner import EvaluationRunner

__all__ = [
    "DeepEvalEvaluator",
    "EvaluationReport",
    "EvaluationRunner",
    "EvaluationSample",
    "MetricResult",
    "RagasEvaluator",
    "ReportWriter",
    "SampleResult",
    "build_metric_registry",
    "build_samples_from_dicts",
    "load_dataset_from_json",
]
