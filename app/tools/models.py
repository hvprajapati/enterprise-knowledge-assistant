"""Data models for the Tool Calling Framework.

These models are the bridge between the agent graph (which needs
serialisable decisions and results) and the tool executor (which
needs structured input/output).
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ToolResult(BaseModel):
    """Captures the outcome of a single tool invocation.

    Attributes
    ----------
    tool_name
        The tool that was invoked (e.g. ``"calculator"``).
    success
        ``True`` when the tool executed without error.
    output
        Primary output — a dict with at least a ``"result"`` key.
    error
        Error message when ``success=False``.
    execution_time_ms
        Wall-clock execution time in milliseconds.
    metadata
        Arbitrary extra data (e.g. tool version, input hashes).
    """

    tool_name: str
    success: bool
    output: dict[str, Any] = Field(default_factory=dict)
    error: str = ""
    execution_time_ms: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)

    # ------------------------------------------------------------------
    # factory helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_success(
        cls,
        *,
        tool_name: str,
        output: dict[str, Any],
        execution_time_ms: float = 0.0,
        metadata: dict[str, Any] | None = None,
    ) -> ToolResult:
        """Create a successful result."""
        return cls(
            tool_name=tool_name,
            success=True,
            output=output,
            execution_time_ms=execution_time_ms,
            metadata=metadata or {},
        )

    @classmethod
    def from_error(
        cls,
        *,
        tool_name: str,
        error: str,
        execution_time_ms: float = 0.0,
        metadata: dict[str, Any] | None = None,
    ) -> ToolResult:
        """Create a failed result."""
        return cls(
            tool_name=tool_name,
            success=False,
            output={},
            error=error,
            execution_time_ms=execution_time_ms,
            metadata=metadata or {},
        )

    def format_for_prompt(self) -> str:
        """Render the tool result as a text block for the LLM prompt.

        The output is a clearly labelled section that the LLM can
        cite alongside retrieved context passages.
        """
        header = f"[Tool: {self.tool_name}]"
        if self.success:
            body = "\n".join(
                f"  {k}: {v}" for k, v in self.output.items()
            )
            return f"{header}\n{body}"
        else:
            return f"{header}\n  ERROR: {self.error}"


class ToolDecision(BaseModel):
    """Decision about whether and which tool to invoke.

    Produced by the tool_node (initially deterministic, later
    replaceable with an LLM-based selector).

    Attributes
    ----------
    use_tool
        ``True`` when a tool should be invoked before the RAG pipeline.
    tool_name
        The name of the selected tool (empty when ``use_tool=False``).
    arguments
        Keyword arguments to pass to ``ToolExecutor.execute()``.
    confidence
        How confident the selector is (0.0–1.0).  Rule-based selectors
        always return 1.0; LLM-based selectors may vary.
    reasoning
        Why this tool was chosen (for logging and debugging).
    """

    use_tool: bool = False
    tool_name: str = ""
    arguments: dict[str, Any] = Field(default_factory=dict)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    reasoning: str = ""

    @classmethod
    def skip_tools(cls) -> ToolDecision:
        """Return a decision that no tool is needed."""
        return cls(
            use_tool=False,
            tool_name="",
            arguments={},
            confidence=1.0,
            reasoning="No tool required for this question type.",
        )

    def to_log_dict(self) -> dict[str, Any]:
        """Compact dict for structured logging."""
        return {
            "use_tool": self.use_tool,
            "tool_name": self.tool_name,
            "confidence": self.confidence,
            "reasoning": self.reasoning[:120],
        }
