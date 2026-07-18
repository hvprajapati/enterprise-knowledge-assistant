"""Unit tests for the Evaluation Framework.

Covers:
- EvaluationSample / MetricResult / EvaluationReport model validation
- build_metric_registry (heuristic fallback always available)
- EvaluationRunner: single sample, dataset, report generation
- RagasEvaluator / DeepEvalEvaluator wrappers
- ReportWriter: JSON and Markdown output
- Error handling: metric failure doesn't stop evaluation
- Dataset loading from dicts
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from app.evaluation import (
    DeepEvalEvaluator,
    EvaluationReport,
    EvaluationRunner,
    EvaluationSample,
    MetricResult,
    RagasEvaluator,
    ReportWriter,
    SampleResult,
    build_metric_registry,
    build_samples_from_dicts,
    load_dataset_from_json,
)

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _make_sample(
    question: str = "What is FAISS?",
    expected: str = "FAISS is a library for similarity search.",
    generated: str = "FAISS is a library for efficient similarity search.",
    context: list[str] | None = None,
) -> EvaluationSample:
    return EvaluationSample(
        question=question,
        expected_answer=expected,
        generated_answer=generated,
        retrieved_context=context or [
            "FAISS is a library for efficient similarity search and clustering.",
            "It supports GPU acceleration and multiple index types.",
        ],
    )


def _simple_metric(sample: EvaluationSample) -> MetricResult:
    """A deterministic metric for testing the runner."""
    score = 0.9 if len(sample.generated_answer) > 10 else 0.2
    return MetricResult(
        metric_name="simple_test",
        score=score,
        passed=score >= 0.6,
        threshold=0.6,
        reasoning=f"Score based on answer length: {score}",
    )


def _failing_metric(sample: EvaluationSample) -> MetricResult:
    raise RuntimeError("Intentional metric failure.")


# ---------------------------------------------------------------------------
# Model tests
# ---------------------------------------------------------------------------


class TestModels:
    def test_evaluation_sample_defaults(self) -> None:
        sample = EvaluationSample(question="Q?")
        assert sample.question == "Q?"
        assert sample.expected_answer == ""
        assert sample.generated_answer == ""
        assert sample.retrieved_context == []

    def test_evaluation_sample_full(self) -> None:
        sample = _make_sample()
        assert "FAISS" in sample.question
        assert len(sample.retrieved_context) == 2

    def test_metric_result_defaults(self) -> None:
        mr = MetricResult(metric_name="test")
        assert mr.score == 0.0
        assert mr.passed is False
        assert mr.threshold == 0.6

    def test_sample_result_failed_metrics(self) -> None:
        sr = SampleResult(
            sample=_make_sample(),
            metrics=[
                MetricResult(metric_name="a", score=0.9, passed=True),
                MetricResult(metric_name="b", score=0.3, passed=False),
            ],
        )
        assert sr.overall_passed is False
        assert len(sr.failed_metrics) == 1

    def test_evaluation_report_defaults(self) -> None:
        report = EvaluationReport()
        assert report.total_samples == 0
        assert report.average_scores == {}


# ---------------------------------------------------------------------------
# Metric registry tests
# ---------------------------------------------------------------------------


class TestMetricRegistry:
    def test_heuristic_metric_always_available(self) -> None:
        registry = build_metric_registry()
        assert "context_relevance_heuristic" in registry

    def test_ragas_metrics_registered(self) -> None:
        registry = build_metric_registry()
        for name in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
            assert name in registry

    def test_deepeval_metrics_registered(self) -> None:
        registry = build_metric_registry()
        for name in ["hallucination", "bias", "toxicity"]:
            assert name in registry

    def test_heuristic_metric_works(self) -> None:
        registry = build_metric_registry()
        metric = registry["context_relevance_heuristic"]
        sample = _make_sample()
        result = metric(sample)
        assert isinstance(result, MetricResult)
        assert result.metric_name == "context_relevance_heuristic"
        assert 0.0 <= result.score <= 1.0


# ---------------------------------------------------------------------------
# EvaluationRunner tests
# ---------------------------------------------------------------------------


class TestEvaluationRunner:
    def test_evaluate_single_sample(self) -> None:
        runner = EvaluationRunner(metrics={"simple": _simple_metric})
        result = runner.evaluate_sample(_make_sample())
        assert result.overall_passed is True
        assert len(result.metrics) == 1
        assert result.metrics[0].metric_name == "simple_test"

    def test_metric_failure_does_not_stop_runner(self) -> None:
        runner = EvaluationRunner(
            metrics={
                "good": _simple_metric,
                "bad": _failing_metric,
            }
        )
        result = runner.evaluate_sample(_make_sample())
        # Should have both metrics, one with error
        assert len(result.metrics) == 2
        errors = [m for m in result.metrics if m.error]
        assert len(errors) == 1

    def test_evaluate_dataset(self) -> None:
        runner = EvaluationRunner(metrics={"simple": _simple_metric})
        samples = [_make_sample() for _ in range(5)]
        results = runner.evaluate_dataset(samples, dataset_name="test_ds")
        assert len(results) == 5

    def test_generate_report(self) -> None:
        runner = EvaluationRunner(metrics={"simple": _simple_metric})
        for _ in range(3):
            runner.evaluate_sample(_make_sample())
        report = runner.generate_report(dataset_name="test_ds")
        assert report.total_samples == 3
        assert report.dataset_name == "test_ds"
        assert "simple_test" in report.average_scores
        assert report.average_scores["simple_test"] > 0.8

    def test_empty_runner_report(self) -> None:
        runner = EvaluationRunner(metrics={})
        report = runner.generate_report()
        assert report.total_samples == 0

    def test_all_metrics_passing_gives_overall_pass(self) -> None:
        def _always_pass(sample: EvaluationSample) -> MetricResult:
            return MetricResult(metric_name="pass", score=0.95, passed=True)

        runner = EvaluationRunner(metrics={"pass": _always_pass})
        result = runner.evaluate_sample(_make_sample())
        assert result.overall_passed is True


# ---------------------------------------------------------------------------
# RagasEvaluator tests
# ---------------------------------------------------------------------------


class TestRagasEvaluator:
    def test_create_evaluator(self) -> None:
        evaluator = RagasEvaluator()
        assert evaluator is not None

    def test_evaluate_sample(self) -> None:
        evaluator = RagasEvaluator(
            thresholds={"faithfulness": 0.7, "answer_relevancy": 0.6,
                        "context_precision": 0.6, "context_recall": 0.6}
        )
        result = evaluator.evaluate(_make_sample())
        assert isinstance(result, SampleResult)
        assert len(result.metrics) == 4  # 4 RAGAS metrics

    def test_evaluate_dataset(self) -> None:
        evaluator = RagasEvaluator()
        samples = [_make_sample() for _ in range(2)]
        results = evaluator.evaluate_dataset(samples, dataset_name="ragas_test")
        assert len(results) == 2

    def test_generate_report(self) -> None:
        evaluator = RagasEvaluator()
        evaluator.evaluate(_make_sample())
        report = evaluator.generate_report(dataset_name="ragas_test")
        assert report.total_samples == 1


# ---------------------------------------------------------------------------
# DeepEvalEvaluator tests
# ---------------------------------------------------------------------------


class TestDeepEvalEvaluator:
    def test_create_evaluator(self) -> None:
        evaluator = DeepEvalEvaluator()
        assert evaluator is not None

    def test_evaluate_sample(self) -> None:
        evaluator = DeepEvalEvaluator(
            thresholds={"hallucination": 0.7, "bias": 0.8, "toxicity": 0.9}
        )
        result = evaluator.evaluate(_make_sample())
        assert isinstance(result, SampleResult)
        assert len(result.metrics) == 5  # 5 DeepEval metrics

    def test_generate_report(self) -> None:
        evaluator = DeepEvalEvaluator()
        evaluator.evaluate(_make_sample())
        report = evaluator.generate_report(dataset_name="de_test")
        assert report.total_samples == 1


# ---------------------------------------------------------------------------
# ReportWriter tests
# ---------------------------------------------------------------------------


class TestReportWriter:
    def test_write_json_and_markdown(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = ReportWriter(output_dir=tmpdir)
            report = EvaluationReport(
                dataset_name="test",
                total_samples=2,
                passed_samples=1,
                failed_samples=1,
                average_scores={"faithfulness": 0.75},
                metric_summaries=[
                    {"metric_name": "faithfulness", "average_score": 0.75, "count": 2}
                ],
            )
            paths = writer.write(report, prefix="test_eval")
            assert Path(paths["json"]).exists()
            assert Path(paths["markdown"]).exists()

            # Verify JSON content
            with open(paths["json"]) as f:
                data = json.load(f)
            assert data["dataset_name"] == "test"
            assert data["average_scores"]["faithfulness"] == 0.75

    def test_format_summary(self) -> None:
        writer = ReportWriter()
        report = EvaluationReport(
            dataset_name="benchmark",
            total_samples=10,
            passed_samples=8,
            failed_samples=2,
            average_scores={"faithfulness": 0.8, "hallucination": 0.65},
            metric_summaries=[],
            recommendations=["Improve context recall."],
        )
        summary = writer.format_summary(report)
        assert "benchmark" in summary
        assert "10 total" in summary
        assert "faithfulness" in summary


# ---------------------------------------------------------------------------
# Dataset loading tests
# ---------------------------------------------------------------------------


class TestDataset:
    def test_build_samples_from_dicts(self) -> None:
        items: list[dict[str, object]] = [
            {"question": "Q1?", "expected_answer": "A1", "generated_answer": "G1",
             "retrieved_context": ["ctx1"]},
            {"question": "Q2?", "expected_answer": "A2"},
        ]
        samples = build_samples_from_dicts(items)
        assert len(samples) == 2
        assert samples[0].question == "Q1?"
        assert samples[0].retrieved_context == ["ctx1"]
        assert samples[1].retrieved_context == []

    def test_load_dataset_from_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "dataset.json"
            data = [
                {
                    "question": "What is FAISS?",
                    "expected_answer": "A library.",
                    "generated_answer": "",
                    "retrieved_context": ["FAISS overview"],
                }
            ]
            with open(path, "w") as f:
                json.dump(data, f)

            samples = load_dataset_from_json(path)
            assert len(samples) == 1
            assert samples[0].question == "What is FAISS?"
            assert samples[0].retrieved_context == ["FAISS overview"]


# ---------------------------------------------------------------------------
# Threshold validation tests
# ---------------------------------------------------------------------------


class TestThresholds:
    def test_metric_passes_when_above_threshold(self) -> None:
        mr = MetricResult(metric_name="test", score=0.85, passed=True, threshold=0.7)
        assert mr.passed is True

    def test_metric_fails_when_below_threshold(self) -> None:
        mr = MetricResult(metric_name="test", score=0.55, passed=False, threshold=0.7)
        assert mr.passed is False

    def test_metric_at_threshold_passes(self) -> None:
        mr = MetricResult(metric_name="test", score=0.7, passed=True, threshold=0.7)
        assert mr.passed is True
