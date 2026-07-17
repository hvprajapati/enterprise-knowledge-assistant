"""Multi-Tool Planner — determines which tools a question needs.

Currently **deterministic** (keyword-based).  The interface is
deliberately narrow so an LLM-based planner can replace it later
without touching the graph or executor.

Design
------
- **Input**: user question + available tool metadata (from registry).
- **Output**: ``ToolExecutionPlan`` with ordered ``ToolInvocation`` list.
- **Deterministic today**: keywords trigger tools (e.g. "calculate" →
  CalculatorTool).  This is fast (microseconds) and predictable.
- **LLM-ready tomorrow**: the ``plan_tools`` signature accepts the
  tool list as dicts, exactly what an LLM prompt would need.

Why plan multiple tools?
    Users ask compound questions: "Summarize the AWS doc, calculate
    monthly cost for 3TB, and tell me today's date."  A single-tool
    selector would pick one and ignore the rest.  Multi-tool planning
    handles all parts of the question in one pass.
"""

from __future__ import annotations

import logging
import re
import time
from typing import Any

from app.tools.planner.models import ToolExecutionPlan, ToolInvocation

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# keyword → tool mapping (deterministic, ordered by priority)
# ---------------------------------------------------------------------------

_KEYWORD_TOOL_MAP: list[tuple[list[str], str]] = [
    (
        ["search", "find documents", "look up", "lookup", "retrieve"],
        "document-search",
    ),
    (
        [
            "calculate", "compute", "evaluate", "math",
            "cost", "price", "total", "sum",
            "multiply", "divide", "subtract", "percentage",
            "how much is", "what is the cost",
        ],
        "calculator",
    ),
    (
        [
            "today", "current date", "current time", "now",
            "what day", "what time", "what is the date",
            "what's the date", "tell me the date",
        ],
        "current-time",
    ),
]

# Additional check: calculator also requires a numeric pattern in the question
_CALC_NUMERIC_RE = re.compile(r"\d+(?:\.\d+)?\s*[\+\-\*\/]")


class MultiToolPlanner:
    """Analyse a question and produce a ``ToolExecutionPlan``.

    Usage::

        planner = MultiToolPlanner(available_tools=registry.list_tools())
        plan = planner.plan_tools(question)
        for invocation in plan.tools:
            print(invocation.tool_name, invocation.arguments)
    """

    def __init__(self, *, available_tools: list[dict[str, Any]] | None = None) -> None:
        self._available = available_tools or []
        self._tool_names = {t["name"] for t in self._available}

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def plan_tools(self, question: str) -> ToolExecutionPlan:
        """Analyse *question* and return an ordered tool plan.

        Parameters
        ----------
        question:
            The raw user question.

        Returns
        -------
        ToolExecutionPlan
            Always returns a plan — empty if no tools are needed.
        """
        t_start = time.monotonic()
        question_lower = question.lower().strip()

        try:
            invocations: list[ToolInvocation] = []
            matched: set[str] = set()

            for keywords, tool_name in _KEYWORD_TOOL_MAP:
                if tool_name in matched:
                    continue
                if tool_name not in self._tool_names:
                    continue
                if any(kw in question_lower for kw in keywords):
                    # Calculator extra guard: must contain numeric expression
                    if tool_name == "calculator" and not _CALC_NUMERIC_RE.search(question_lower):
                        continue
                    invocations.append(
                        ToolInvocation(
                            tool_name=tool_name,
                            arguments=self._build_arguments(tool_name, question),
                            optional=tool_name != "document-search",
                        )
                    )
                    matched.add(tool_name)

            if not invocations:
                plan = ToolExecutionPlan.empty()
            else:
                reasoning = (
                    f"Matched {len(invocations)} tool(s) for question: "
                    + ", ".join(t.tool_name for t in invocations)
                )
                plan = ToolExecutionPlan(
                    tools=invocations,
                    reasoning=reasoning,
                    expected_outputs=[
                        f"{t.tool_name}: result" for t in invocations
                    ],
                )

            elapsed = (time.monotonic() - t_start) * 1000
            logger.info(
                "MultiToolPlanner — tools=%d  names=%s  latency=%.0fms",
                plan.tool_count,
                [t.tool_name for t in plan.tools],
                elapsed,
            )
            return plan

        except Exception:
            logger.exception("MultiToolPlanner failed — returning empty plan")
            return ToolExecutionPlan.empty()

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_arguments(tool_name: str, question: str) -> dict[str, Any]:
        """Extract tool arguments from the question text.

        For calculator: try to extract a numeric expression.
        For document-search: use the full question as query.
        For current-time: no arguments needed.
        """
        if tool_name == "calculator":
            expr = _extract_expression(question)
            return {"expression": expr} if expr else {"expression": question}
        elif tool_name == "document-search":
            return {"query": question, "top_k": 5}
        elif tool_name == "current-time":
            return {}
        return {"question": question}


# ---------------------------------------------------------------------------
# expression extraction
# ---------------------------------------------------------------------------

_EXPRESSION_PATTERNS = [
    re.compile(r"(\d+(?:\.\d+)?\s*[\+\-\*\/\*\*]\s*\d+(?:\.\d+)?(?:\s*[\+\-\*\/\*\*]\s*\d+(?:\.\d+)?)*)"),
    re.compile(r"(\d+(?:\.\d+)?\s*[\+\-\*\/]\s*\d+(?:\.\d+)?)"),
]


def _extract_expression(question: str) -> str | None:
    """Try to pull a mathematical expression out of the question text."""
    for pattern in _EXPRESSION_PATTERNS:
        match = pattern.search(question)
        if match:
            return match.group(1).strip()
    return None
