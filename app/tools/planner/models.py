"""Data models for multi-tool execution planning.

These models describe *what* tools to run, in *what order*, and with
*what dependencies*.  They are produced by ``MultiToolPlanner`` and
consumed by ``SequentialToolExecutor``.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ToolInvocation(BaseModel):
    """A single tool call within a ``ToolExecutionPlan``.

    Attributes
    ----------
    tool_name
        Registered tool identifier (e.g. ``"calculator"``).
    arguments
        Keyword arguments to pass to ``BaseTool.invoke()``.
    optional
        When ``True``, failure of this tool does NOT abort the plan.
        Required tools (``optional=False``) abort the plan on failure.
    depends_on
        Tool name(s) this invocation must wait for.  Currently unused
        (sequential execution only) but reserved for future parallel
        execution support.
    """

    tool_name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    optional: bool = False
    depends_on: list[str] = Field(default_factory=list)


class ToolExecutionPlan(BaseModel):
    """Ordered sequence of tool invocations.

    Produced by ``MultiToolPlanner.plan_tools()`` and stored in
    ``AgentState.tool_execution_plan``.

    Attributes
    ----------
    tools
        Ordered list of ``ToolInvocation`` to execute.
    reasoning
        Why these tools were chosen and in this order.
    expected_outputs
        Human-readable summary of what each tool should produce
        (useful for prompt building and logging).
    """

    tools: list[ToolInvocation] = Field(default_factory=list)
    reasoning: str = ""
    expected_outputs: list[str] = Field(default_factory=list)

    # ------------------------------------------------------------------
    # queries
    # ------------------------------------------------------------------

    @property
    def tool_count(self) -> int:
        """Total number of invocations in the plan."""
        return len(self.tools)

    @property
    def required_tools(self) -> list[ToolInvocation]:
        """Non-optional invocations (plan aborts if any fail)."""
        return [t for t in self.tools if not t.optional]

    @property
    def optional_tools(self) -> list[ToolInvocation]:
        """Optional invocations (plan continues on failure)."""
        return [t for t in self.tools if t.optional]

    def is_empty(self) -> bool:
        """``True`` when no tools are planned."""
        return len(self.tools) == 0

    # ------------------------------------------------------------------
    # factory
    # ------------------------------------------------------------------

    @classmethod
    def empty(cls) -> ToolExecutionPlan:
        """Return a plan that skips all tool execution."""
        return cls(tools=[], reasoning="No tools required for this question.")

    def to_log_dict(self) -> dict[str, Any]:
        """Compact dict for structured logging."""
        return {
            "tool_count": self.tool_count,
            "tool_names": [t.tool_name for t in self.tools],
            "required_count": len(self.required_tools),
            "optional_count": len(self.optional_tools),
            "reasoning": self.reasoning[:120],
        }
