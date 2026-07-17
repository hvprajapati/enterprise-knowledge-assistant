"""Validation Agent — gates the answer against quality thresholds.

Wraps the existing ``AnswerValidator``.
"""

from __future__ import annotations

from typing import Any

from app.agent.reflection import ReflectionResult
from app.agent.state import AgentState
from app.agent.validation import ValidationResult
from app.agent.validation.validator import AnswerValidator
from app.agents.base import BaseAgent


class ValidationAgent(BaseAgent):
    """Apply quality thresholds to the ReflectionResult.

    Requires the ``validator`` service to be registered.
    """

    @property
    def name(self) -> str:
        return "validation"

    @property
    def description(self) -> str:
        return "Gates the answer against configurable quality thresholds."

    def execute(self, state: AgentState) -> dict[str, Any]:
        from app.agent.nodes import _get_service

        reflection_dict = state.get("reflection_result", {})
        reflection = ReflectionResult.model_validate(reflection_dict)

        validator: AnswerValidator = _get_service("validator")
        result: ValidationResult = validator.validate(reflection)

        executed = list(state.get("executed_nodes", [])) + ["validation"]
        history = list(state.get("execution_history", [])) + [
            {
                "agent": self.name,
                "outcome": "success",
                "passed": result.passed,
                "severity": result.severity.value,
            }
        ]

        return {
            "validation_result": result.model_dump(),
            "executed_nodes": executed,
            "execution_history": history,
            "current_agent": self.name,
        }
