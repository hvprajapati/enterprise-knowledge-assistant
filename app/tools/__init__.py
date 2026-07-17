"""Generic Tool Calling Framework for the Enterprise Knowledge Agent.

The framework provides:

- ``BaseTool`` — abstract base class for all tools.
- ``ToolRegistry`` — in-memory catalogue of available tools.
- ``ToolExecutor`` — safe, measured tool invocation.
- ``ToolResult`` / ``ToolDecision`` — structured Pydantic models.

Adding a new tool
-----------------
1. Subclass ``BaseTool`` and implement ``name``, ``description``,
   ``schema``, and ``invoke()``.
2. Register it: ``registry.register(MyTool())``.
3. The planner and tool_node will pick it up automatically.

No graph changes needed — the framework is fully generic.
"""

from __future__ import annotations

from app.tools.base import BaseTool
from app.tools.executor import ToolExecutor
from app.tools.models import ToolDecision, ToolResult
from app.tools.registry import ToolRegistry

__all__ = [
    "BaseTool",
    "ToolDecision",
    "ToolExecutor",
    "ToolRegistry",
    "ToolResult",
]
