"""Remote Tool Adapter — wraps remote MCP tools as local ``BaseTool``.

This is the key integration point.  The rest of the application
(``ToolRegistry``, ``ToolExecutor``, multi-tool planner) works with
``BaseTool`` instances.  ``RemoteToolAdapter`` implements the
``BaseTool`` interface using remote MCP calls, so the app never
needs to know whether a tool is local or remote.

Design
------
- **Same interface.**  ``RemoteToolAdapter`` subclasses ``BaseTool``
  and implements ``invoke(**kwargs)`` by calling the remote MCP
  server synchronously (via ``asyncio.run`` internally).
- **Cached metadata.**  ``name``, ``description``, and ``schema``
  come from the MCP ``Tool`` object discovered at connection time.
- **Lazy connection.**  The adapter holds a reference to the
  ``MCPClient``; the actual connection lifecycle is managed by
  ``MCPConnectionManager``.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from app.mcp.client.models import RemoteToolInfo
from app.tools.base import BaseTool

logger = logging.getLogger(__name__)


class RemoteToolAdapter(BaseTool):
    """Wraps a tool on a remote MCP server as a local ``BaseTool``.

    Usage::

        adapter = RemoteToolAdapter(
            tool_info=remote_tool_info,
            mcp_client=mcp_client,
        )
        result = adapter.invoke(expression="2 + 2")
        # → {"result": 4}
    """

    def __init__(
        self,
        *,
        tool_info: RemoteToolInfo,
        mcp_client: Any,  # MCPClient (avoid circular import)
    ) -> None:
        self._tool_info = tool_info
        self._client = mcp_client

    # ------------------------------------------------------------------
    # BaseTool interface
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return self._tool_info.name

    @property
    def description(self) -> str:
        return self._tool_info.description

    @property
    def schema(self) -> dict[str, Any]:
        return self._tool_info.input_schema

    def invoke(self, **kwargs: Any) -> dict[str, Any]:
        """Call the remote MCP tool synchronously.

        Uses ``asyncio.run`` to bridge the async MCP SDK into the
        synchronous ``BaseTool.invoke()`` contract.  This is safe
        because tool invocations are called from a thread pool in
        the tool executor, not from an async context.
        """
        try:
            return asyncio.run(self._client.call_tool(self.name, kwargs))
        except Exception as exc:
            logger.exception(
                "Remote tool '%s' on server '%s' failed: %s",
                self.name,
                self._client.server_name,
                exc,
            )
            raise RuntimeError(
                f"Remote tool '{self.name}' failed: {exc}"
            ) from exc
