"""Tool Executor — validates and invokes tools safely.

The executor is the **only** code path that runs tools.  Every
invocation goes through ``execute()``, which:

1. Looks up the tool in the registry (fail-fast if missing).
2. Validates arguments against the tool's JSON Schema (shallow check).
3. Invokes the tool with structured error handling.
4. Wraps the result in a ``ToolResult`` with execution metadata.

Design
------
- **Single choke-point.**  All tool calls flow through here — logging,
  metrics, and security checks only need to be added in one place.
- **Fail-safe.**  A tool that raises an exception returns a
  ``ToolResult(success=False)`` instead of crashing the graph.
- **Measured.**  Every invocation includes wall-clock latency.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from app.tools.base import BaseTool
from app.tools.models import ToolResult
from app.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


class ToolExecutor:
    """Safe, measured tool invocation through a registry.

    Usage::

        registry = ToolRegistry()
        registry.register(CalculatorTool())
        executor = ToolExecutor(registry=registry)

        result = executor.execute("calculator", {"expression": "2 + 2"})
        if result.success:
            print(result.output["result"])  # 4
    """

    def __init__(self, *, registry: ToolRegistry) -> None:
        self._registry = registry

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def execute(
        self,
        tool_name: str,
        arguments: dict[str, Any] | None = None,
    ) -> ToolResult:
        """Validate and invoke a tool.

        Parameters
        ----------
        tool_name:
            The name of a registered tool (e.g. ``"calculator"``).
        arguments:
            Keyword arguments to pass to ``BaseTool.invoke()``.
            Defaults to empty dict.

        Returns
        -------
        ToolResult
            Always returns a result — on failure returns
            ``ToolResult.from_error(...)``.
        """
        kwargs = arguments or {}
        t_start = time.monotonic()

        logger.info(
            "Tool execution requested — tool=%s  args=%s",
            tool_name,
            {k: str(v)[:60] for k, v in kwargs.items()},
        )

        # 1. Lookup
        tool = self._registry.lookup(tool_name)
        if tool is None:
            elapsed = (time.monotonic() - t_start) * 1000
            logger.error(
                "Tool not found: %s  latency=%.0fms",
                tool_name,
                elapsed,
            )
            return ToolResult.from_error(
                tool_name=tool_name,
                error=f"Tool '{tool_name}' is not registered. "
                f"Available: {[t['name'] for t in self._registry.list_tools()]}",
                execution_time_ms=elapsed,
            )

        # 2. Validate arguments (basic: required keys present)
        validation_error = self._validate_arguments(tool, kwargs)
        if validation_error:
            elapsed = (time.monotonic() - t_start) * 1000
            logger.warning(
                "Tool argument validation failed — tool=%s  error=%s",
                tool_name,
                validation_error,
            )
            return ToolResult.from_error(
                tool_name=tool_name,
                error=validation_error,
                execution_time_ms=elapsed,
            )

        # 3. Invoke
        try:
            output = tool.invoke(**kwargs)
            elapsed = (time.monotonic() - t_start) * 1000

            logger.info(
                "Tool succeeded — tool=%s  latency=%.0fms",
                tool_name,
                elapsed,
            )
            return ToolResult.from_success(
                tool_name=tool_name,
                output=output,
                execution_time_ms=elapsed,
            )

        except Exception as exc:
            elapsed = (time.monotonic() - t_start) * 1000
            logger.exception(
                "Tool failed — tool=%s  error=%s  latency=%.0fms",
                tool_name,
                exc,
                elapsed,
            )
            return ToolResult.from_error(
                tool_name=tool_name,
                error=str(exc),
                execution_time_ms=elapsed,
            )

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_arguments(
        tool: BaseTool,
        arguments: dict[str, Any],
    ) -> str:
        """Check that required schema keys are present.

        Returns an error string, or empty string if valid.
        """
        schema = tool.schema
        required: list[str] = schema.get("required", [])
        properties: dict[str, Any] = schema.get("properties", {})

        # Check required keys
        missing = [k for k in required if k not in arguments]
        if missing:
            return (
                f"Missing required arguments for '{tool.name}': {missing}. "
                f"Schema requires: {required}"
            )

        # Check type of each provided argument (best-effort)
        for key, value in arguments.items():
            if key in properties:
                expected_type = properties[key].get("type", "string")
                if not _type_matches(value, expected_type):
                    return (
                        f"Type mismatch for argument '{key}': "
                        f"expected {expected_type}, got {type(value).__name__}"
                    )

        return ""


def _type_matches(value: object, json_type: str) -> bool:
    """Best-effort Python ↔ JSON Schema type check."""
    mapping: dict[str, type | tuple[type, ...]] = {
        "string": str,
        "number": (int, float),
        "integer": int,
        "boolean": bool,
        "object": dict,
        "array": list,
    }
    expected = mapping.get(json_type)
    if expected is None:
        return True  # unknown types pass
    return isinstance(value, expected)
