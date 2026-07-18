"""Dataset utilities for the evaluation framework.

Load evaluation samples from JSON files, Python lists, or
programmatic construction.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from app.evaluation.models import EvaluationSample

logger = logging.getLogger(__name__)


def load_dataset_from_json(path: str | Path) -> list[EvaluationSample]:
    """Load evaluation samples from a JSON file.

    Expected format::

        [
            {
                "question": "What is FAISS?",
                "expected_answer": "FAISS is a library for similarity search.",
                "generated_answer": "",
                "retrieved_context": ["passage 1", "passage 2"],
                "metadata": {}
            },
            ...
        ]

    ``generated_answer`` and ``retrieved_context`` may be empty —
    the runner will populate them from the RAG pipeline if needed.
    """
    path = Path(path)
    logger.info("Loading dataset: %s", path)

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    samples = []
    for i, item in enumerate(data):
        try:
            samples.append(
                EvaluationSample(
                    question=str(item.get("question", "")),
                    expected_answer=str(item.get("expected_answer", "")),
                    generated_answer=str(item.get("generated_answer", "")),
                    retrieved_context=[
                        str(c)
                        for c in item.get("retrieved_context", [])
                    ],
                    metadata=item.get("metadata", {}) or {},
                )
            )
        except Exception as exc:
            logger.warning("Skipping sample %d: %s", i, exc)

    logger.info("Loaded %d samples from %s", len(samples), path)
    return samples


def build_samples_from_dicts(
    items: list[dict[str, Any]],
) -> list[EvaluationSample]:
    """Build ``EvaluationSample`` objects from a list of dicts.

    Useful for programmatic dataset construction in tests and scripts.
    """
    samples: list[EvaluationSample] = []
    for item in items:
        samples.append(
            EvaluationSample(
                question=str(item.get("question", "")),
                expected_answer=str(item.get("expected_answer", "")),
                generated_answer=str(item.get("generated_answer", "")),
                retrieved_context=[
                    str(c) for c in item.get("retrieved_context", [])
                ],
                metadata=item.get("metadata", {}) or {},
            )
        )
    return samples
