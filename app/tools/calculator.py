"""Calculator tool — evaluates simple mathematical expressions.

This is a **placeholder** tool for demonstration.  In production it
could be replaced with a sandboxed Python evaluator or a symbolic
math engine.  For now it uses Python's built-in ``eval()`` with a
restricted namespace (no builtins, only operators and ``math``).
"""

from __future__ import annotations

import math
import operator
from typing import Any

from app.tools.base import BaseTool


class CalculatorTool(BaseTool):
    """Evaluate a mathematical expression and return the numeric result."""

    @property
    def name(self) -> str:
        return "calculator"

    @property
    def description(self) -> str:
        return (
            "Evaluate a mathematical expression. "
            "Supports basic arithmetic (+, -, *, /, **), "
            "parentheses, and common math functions (sqrt, sin, cos, log, abs). "
            "Use when the user asks for a calculation or numeric computation."
        )

    @property
    def schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": (
                        "Mathematical expression to evaluate, e.g. '2 + 2 * 3' "
                        "or 'sqrt(16) + abs(-5)'."
                    ),
                },
            },
            "required": ["expression"],
        }

    def invoke(self, **kwargs: Any) -> dict[str, Any]:
        expression = str(kwargs.get("expression", ""))
        if not expression.strip():
            raise ValueError("Expression is empty.")

        # Restricted namespace: only safe math functions, no builtins
        namespace: dict[str, Any] = {
            name: getattr(math, name)
            for name in dir(math)
            if not name.startswith("_")
        }
        namespace.update(vars(operator))

        try:
            result = eval(expression, {"__builtins__": {}}, namespace)
        except SyntaxError as exc:
            raise ValueError(f"Invalid expression syntax: {exc}") from exc
        except NameError as exc:
            raise ValueError(
                f"Unknown name in expression: {exc}. "
                f"Only math functions and basic operators are supported."
            ) from exc
        except Exception as exc:
            raise ValueError(f"Failed to evaluate expression: {exc}") from exc

        return {
            "result": result,
            "expression": expression,
        }
