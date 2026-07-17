"""Agent Validation — deterministic quality gate after reflection.

The validation sub-package inspects the ``ReflectionResult`` and
decides whether the answer meets minimum quality thresholds.
It is a **read-only gate** — it never modifies the answer or the
retrieval pipeline.

Components
----------
- ``ValidationResult`` — structured Pydantic model for the gate decision.
- ``AnswerValidator`` — deterministic rule-based validator.
- ``validation_node`` — LangGraph node (in ``app.agent.nodes``).
"""

from __future__ import annotations

from app.agent.validation.models import (
    ValidationCheck,
    ValidationResult,
    ValidationSeverity,
)
from app.agent.validation.validator import AnswerValidator, ValidationThresholds

__all__ = [
    "AnswerValidator",
    "ValidationCheck",
    "ValidationResult",
    "ValidationSeverity",
    "ValidationThresholds",
]
