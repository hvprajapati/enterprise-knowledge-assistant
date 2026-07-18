"""Report generation — JSON and Markdown output.

Produces human-readable and machine-readable evaluation reports
from ``EvaluationReport`` objects.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.evaluation.models import EvaluationReport

logger = logging.getLogger(__name__)


class ReportWriter:
    """Write ``EvaluationReport`` to disk in JSON and Markdown formats.

    Usage::

        writer = ReportWriter(output_dir="reports/")
        writer.write(report, prefix="rag_eval_v1")
    """

    def __init__(self, output_dir: str = "reports") -> None:
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def write(
        self,
        report: EvaluationReport,
        *,
        prefix: str = "evaluation",
    ) -> dict[str, str]:
        """Write JSON and Markdown reports.

        Returns
        -------
        dict[str, str]
            Paths to the generated files (``json``, ``markdown``).
        """
        ts = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        base = f"{prefix}_{ts}"

        json_path = str(self._output_dir / f"{base}.json")
        md_path = str(self._output_dir / f"{base}.md")

        self._write_json(report, json_path)
        self._write_markdown(report, md_path)

        logger.info("Reports written: %s, %s", json_path, md_path)
        return {"json": json_path, "markdown": md_path}

    # ------------------------------------------------------------------
    # formatters
    # ------------------------------------------------------------------

    def format_summary(self, report: EvaluationReport) -> str:
        """Return a human-readable summary string."""
        lines = [
            f"Evaluation Report: {report.dataset_name or '(unnamed)'}",
            f"Timestamp: {report.timestamp.isoformat()}",
            f"Samples: {report.total_samples} total, "
            f"{report.passed_samples} passed, {report.failed_samples} failed",
            f"Total latency: {report.total_latency_ms:.0f}ms",
            "",
            "Metric Averages:",
        ]
        for name, score in report.average_scores.items():
            status = "+" if score >= 0.6 else "-"
            lines.append(f"  [{status}] {name}: {score:.3f}")
        if report.recommendations:
            lines.append("\nRecommendations:")
            for rec in report.recommendations:
                lines.append(f"  • {rec}")
        return "\n".join(lines)

    def _format_markdown(self, report: EvaluationReport) -> str:
        """Build a full Markdown report."""
        lines = [
            f"# Evaluation Report: {report.dataset_name or '(unnamed)'}",
            "",
            f"**Timestamp:** {report.timestamp.isoformat()}",
            "",
            "## Summary",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Total Samples | {report.total_samples} |",
            f"| Passed | {report.passed_samples} |",
            f"| Failed | {report.failed_samples} |",
            f"| Pass Rate | "
            f"{report.passed_samples / max(report.total_samples, 1) * 100:.1f}% |",
            f"| Total Latency | {report.total_latency_ms:.0f}ms |",
            "",
            "## Metric Averages",
            "",
            "| Metric | Average Score | Pass/Fail |",
            "|--------|---------------|-----------|",
        ]
        for summary in report.metric_summaries:
            avg = summary["average_score"]
            status = "✅" if avg >= 0.6 else "❌"
            lines.append(
                f"| {summary['metric_name']} | {avg:.3f} | {status} |"
            )

        if report.failures:
            lines.append("")
            lines.append("## Failed Samples")
            lines.append("")
            for i, f in enumerate(report.failures, 1):
                lines.append(f"### {i}. {f['question']}")
                lines.append(f"Score: {f['overall_score']:.3f}")
                lines.append("")
                for fm in f["failed_metrics"]:
                    lines.append(f"- **{fm['name']}**: {fm['score']:.3f}")
                    if fm["reasoning"]:
                        lines.append(f"  > {fm['reasoning'][:200]}")

        if report.recommendations:
            lines.append("")
            lines.append("## Recommendations")
            for rec in report.recommendations:
                lines.append(f"- {rec}")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # internal
    # ------------------------------------------------------------------

    def _write_json(self, report: EvaluationReport, path: str) -> None:
        data: dict[str, Any] = {
            "dataset_name": report.dataset_name,
            "timestamp": report.timestamp.isoformat(),
            "total_samples": report.total_samples,
            "passed_samples": report.passed_samples,
            "failed_samples": report.failed_samples,
            "average_scores": report.average_scores,
            "metric_summaries": report.metric_summaries,
            "failures": report.failures,
            "recommendations": report.recommendations,
            "evaluator_errors": report.evaluator_errors,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)
        logger.info("JSON report written: %s", path)

    def _write_markdown(self, report: EvaluationReport, path: str) -> None:
        md = self._format_markdown(report)
        with open(path, "w", encoding="utf-8") as f:
            f.write(md)
        logger.info("Markdown report written: %s", path)
