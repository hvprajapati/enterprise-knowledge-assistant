"""MCP Tool Adapter — converts BaseTool instances into MCP-compatible tools.

Design
------
Every ``BaseTool`` in the application already has ``name``,
``description``, ``schema``, and ``invoke()``.  The MCP protocol
expects the same metadata plus a handler that returns
``list[TextContent]``.

``MCPToolAdapter`` wraps a ``BaseTool`` and provides an ``invoke()``
method that calls the underlying tool and formats the result as MCP
``TextContent``.  This avoids duplicating any tool implementation.

Why adapters?
    - **No duplication.**  Tools are defined once (``BaseTool``).
      The adapter translates between app types and MCP types.
    - **Separation of concerns.**  The MCP layer knows nothing about
      ``CalculatorTool`` internals — it only sees ``BaseTool``.
    - **Auto-discovery.**  Every registered ``BaseTool`` can be
      automatically exposed as an MCP tool via the adapter.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from mcp.types import TextContent

from app.tools.base import BaseTool

logger = logging.getLogger(__name__)


class MCPToolAdapter:
    """Wrap a ``BaseTool`` for MCP-compatible invocation.

    Usage::

        adapter = MCPToolAdapter(calculator_tool)
        # MCP discovery
        tool_meta = adapter.to_mcp_tool()  # → mcp.types.Tool
        # MCP invocation
        content = await adapter.invoke({"expression": "2 + 2"})
    """

    def __init__(self, tool: BaseTool) -> None:
        self._tool = tool

    # ------------------------------------------------------------------
    # properties
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return self._tool.name

    @property
    def description(self) -> str:
        return self._tool.description

    @property
    def schema(self) -> dict[str, Any]:
        return self._tool.schema

    # ------------------------------------------------------------------
    # MCP tool discovery
    # ------------------------------------------------------------------

    def to_mcp_tool(self) -> Any:
        """Return an MCP ``Tool`` object for the ``list_tools`` handler.

        Import is deferred to avoid coupling the whole app to MCP types
        at import time.
        """
        from mcp.types import Tool

        return Tool(
            name=self.name,
            description=self.description,
            inputSchema=self.schema,
        )

    # ------------------------------------------------------------------
    # MCP invocation
    # ------------------------------------------------------------------

    def invoke(self, arguments: dict[str, Any]) -> list[TextContent]:
        """Execute the underlying tool and return MCP content.

        Parameters
        ----------
        arguments:
            Keyword arguments matching the tool's JSON Schema.

        Returns
        -------
        list[TextContent]
            Always returns content — on error includes the error text.
        """
        t_start = time.monotonic()
        logger.info("MCP tool invoked: %s  args=%s", self.name, arguments)

        try:
            output = self._tool.invoke(**arguments)
            elapsed = (time.monotonic() - t_start) * 1000

            text = json.dumps(output, indent=2, default=str)
            logger.info(
                "MCP tool success: %s  latency=%.0fms", self.name, elapsed
            )
            return [TextContent(type="text", text=text)]

        except Exception as exc:
            elapsed = (time.monotonic() - t_start) * 1000
            logger.exception(
                "MCP tool failed: %s  error=%s  latency=%.0fms",
                self.name,
                exc,
                elapsed,
            )
            return [
                TextContent(
                    type="text",
                    text=json.dumps({
                        "error": str(exc),
                        "tool": self.name,
                    }),
                )
            ]
