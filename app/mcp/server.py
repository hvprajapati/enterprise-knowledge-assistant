"""MCP Server — exposes the Enterprise Knowledge Assistant as an MCP tool provider.

This server follows the official MCP specification and uses the
``mcp`` Python SDK.  It exposes all registered ``BaseTool`` instances
as MCP tools that clients (Claude Desktop, Claude Code, Cursor, etc.)
can discover and invoke.

Architecture
------------
::

    MCP Client (Claude Desktop)
         │
         ▼
    StdioServerTransport (stdin/stdout JSON-RPC)
         │
         ▼
    mcp.server.Server
         ├── list_tools handler → MCPRegistry.list_tools()
         └── call_tool handler  → MCPToolAdapter.invoke()

Usage
-----
::

    import asyncio
    from app.mcp.server import MCPServer

    server = MCPServer(mcp_registry=registry, config=config)
    asyncio.run(server.serve())
"""

from __future__ import annotations

import logging
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    TextContent,
)

from app.mcp.models import MCPServerConfig
from app.mcp.registry import MCPRegistry

logger = logging.getLogger(__name__)


class MCPServer:
    """MCP-compliant server wrapping the Enterprise Knowledge Assistant.

    Parameters
    ----------
    mcp_registry:
        The MCP tool registry with pre-built adapters.
    config:
        Server configuration (name, version, enable flag).
    """

    def __init__(
        self,
        *,
        mcp_registry: MCPRegistry,
        config: MCPServerConfig | None = None,
    ) -> None:
        self._registry = mcp_registry
        self._config = config or MCPServerConfig()

        # Build the underlying MCP SDK server
        self._server = Server(self._config.name)

        # Register handlers
        self._register_handlers()

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    async def serve(self) -> None:
        """Start the MCP server on stdio transport.

        This method blocks until the client disconnects or the process
        is terminated.  It uses stdin/stdout for JSON-RPC communication
        as specified by the MCP protocol.
        """
        if not self._config.enable:
            logger.info("MCP server is disabled by configuration.")
            return

        logger.info(
            "MCP server starting — name=%s  version=%s  tools=%d",
            self._config.name,
            self._config.version,
            self._registry.tool_count,
        )

        # Sync tools from the app ToolRegistry to MCP adapters
        self._registry.sync()

        async with stdio_server() as (read_stream, write_stream):
            init_opts = self._server.create_initialization_options()
            await self._server.run(
                read_stream,
                write_stream,
                init_opts,
                raise_exceptions=False,
            )

        logger.info("MCP server stopped.")

    # ------------------------------------------------------------------
    # handler registration
    # ------------------------------------------------------------------

    def _register_handlers(self) -> None:
        """Register list_tools and call_tool handlers on the MCP server."""

        # -- tool discovery ---------------------------------------------
        @self._server.list_tools()
        async def handle_list_tools() -> list[Any]:  # pragma: no cover
            logger.info("MCP list_tools — %d tools available", self._registry.tool_count)
            return self._registry.list_tools()

        # -- tool invocation --------------------------------------------
        @self._server.call_tool()
        async def handle_call_tool(
            name: str, arguments: dict[str, Any]
        ) -> list[TextContent]:  # pragma: no cover
            logger.info("MCP call_tool — tool=%s", name)

            adapter = self._registry.lookup(name)
            if adapter is None:
                logger.warning("MCP call_tool — unknown tool: %s", name)
                available = list(self._registry._adapters.keys())
                return [
                    TextContent(
                        type="text",
                        text=(
                            f"Error: Unknown tool '{name}'. "
                            f"Available: {available}"
                        ),
                    )
                ]

            return adapter.invoke(arguments)


# ---------------------------------------------------------------------------
# convenience entry point
# ---------------------------------------------------------------------------


def create_mcp_server(
    *,
    mcp_registry: MCPRegistry,
    config: MCPServerConfig | None = None,
) -> MCPServer:
    """Factory for ``MCPServer`` — useful for script entry points."""
    return MCPServer(mcp_registry=mcp_registry, config=config)
