"""MCP Client — connects to a single external MCP server.

Wraps the MCP SDK's ``ClientSession`` and ``stdio_client`` transport
into a simpler interface for the application layer.

Design
------
- **Single-server.**  One ``MCPClient`` per external MCP server.
  ``MCPConnectionManager`` manages multiple clients.
- **Async internally, sync externally.**  The MCP SDK is async, but
  the application's tool interface is synchronous.  ``call_tool``
  bridges the gap via ``asyncio.run`` when invoked from non-async code.
- **Graceful degradation.**  Connection failures are captured as
  ``ConnectionStatus.ERROR`` — the application never crashes because
  a remote server is down.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

from app.mcp.client.models import ConnectionStatus, MCPServerConnection, RemoteToolInfo

logger = logging.getLogger(__name__)


class MCPClient:
    """Connect to and interact with a single external MCP server.

    Usage::

        conn = MCPServerConnection(name="math", command="python", args=["-m", "math_server"])
        client = MCPClient(connection=conn)
        await client.connect()
        tools = await client.discover_tools()
        result = await client.call_tool("add", {"a": 1, "b": 2})
        await client.disconnect()
    """

    def __init__(self, connection: MCPServerConnection) -> None:
        self._conn = connection
        self._session: ClientSession | None = None
        self._read_stream = None
        self._write_stream = None

    # ------------------------------------------------------------------
    # properties
    # ------------------------------------------------------------------

    @property
    def server_name(self) -> str:
        return self._conn.name

    @property
    def status(self) -> ConnectionStatus:
        return self._conn.status

    @property
    def tools(self) -> list[RemoteToolInfo]:
        return list(self._conn.tools)

    # ------------------------------------------------------------------
    # connection lifecycle
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        """Establish a connection to the remote MCP server.

        Spawns the server process (via ``StdioServerParameters``),
        performs the MCP handshake, and caches the session.
        """
        t_start = time.monotonic()
        self._conn.status = ConnectionStatus.CONNECTING
        logger.info("MCP client connecting to '%s' ...", self._conn.name)

        try:
            params = StdioServerParameters(
                command=self._conn.command,
                args=self._conn.args,
                env=self._conn.env,
            )

            # stdio_client is an async context manager
            self._ctx = stdio_client(params)
            read_stream, write_stream = await self._ctx.__aenter__()

            self._session = ClientSession(read_stream, write_stream)
            await self._session.initialize()

            self._conn.status = ConnectionStatus.CONNECTED
            elapsed = (time.monotonic() - t_start) * 1000
            logger.info(
                "MCP client connected to '%s'  latency=%.0fms",
                self._conn.name,
                elapsed,
            )

        except Exception as exc:
            self._conn.status = ConnectionStatus.ERROR
            self._conn.last_error = str(exc)
            elapsed = (time.monotonic() - t_start) * 1000
            logger.error(
                "MCP client failed to connect to '%s': %s  latency=%.0fms",
                self._conn.name,
                exc,
                elapsed,
            )
            raise

    async def disconnect(self) -> None:
        """Gracefully close the connection."""
        logger.info("MCP client disconnecting from '%s' ...", self._conn.name)
        try:
            if hasattr(self, "_ctx"):
                await self._ctx.__aexit__(None, None, None)
        except Exception as exc:
            logger.warning("Error during disconnect from '%s': %s", self._conn.name, exc)
        finally:
            self._session = None
            self._conn.status = ConnectionStatus.DISCONNECTED
        logger.info("MCP client disconnected from '%s'", self._conn.name)

    # ------------------------------------------------------------------
    # tool operations
    # ------------------------------------------------------------------

    async def discover_tools(self) -> list[RemoteToolInfo]:
        """Fetch the tool list from the server and cache it."""
        if self._session is None:
            raise RuntimeError(f"Not connected to '{self._conn.name}'")

        logger.info("Discovering tools on '%s' ...", self._conn.name)
        result = await self._session.list_tools()

        tools = [
            RemoteToolInfo(
                name=t.name,
                description=t.description or "",
                input_schema=t.inputSchema,
                server_name=self._conn.name,
            )
            for t in result.tools
        ]

        self._conn.tools = tools
        logger.info(
            "Discovered %d tool(s) on '%s': %s",
            len(tools),
            self._conn.name,
            [t.name for t in tools],
        )
        return tools

    async def call_tool(
        self, tool_name: str, arguments: dict[str, Any]
    ) -> dict[str, Any]:
        """Invoke a tool on the remote server and return the result dict.

        Parameters
        ----------
        tool_name:
            Name of the remote tool to call.
        arguments:
            Keyword arguments matching the tool's input schema.

        Returns
        -------
        dict[str, Any]
            The structured result from the tool.  MCP content blocks
            are merged into a single dict.

        Raises
        ------
        RuntimeError
            If not connected or the tool call fails.
        """
        if self._session is None:
            raise RuntimeError(f"Not connected to '{self._conn.name}'")

        t_start = time.monotonic()
        logger.info(
            "Calling remote tool '%s' on '%s'  args=%s",
            tool_name,
            self._conn.name,
            {k: str(v)[:60] for k, v in arguments.items()},
        )

        try:
            result = await self._session.call_tool(tool_name, arguments)
            elapsed = (time.monotonic() - t_start) * 1000

            # Merge content blocks into a single result dict
            import json
            output: dict[str, Any] = {}
            for block in result.content:
                if hasattr(block, "text"):
                    try:
                        output.update(json.loads(block.text))
                    except (json.JSONDecodeError, TypeError):
                        output["text"] = block.text

            logger.info(
                "Remote tool '%s' on '%s' succeeded  latency=%.0fms",
                tool_name,
                self._conn.name,
                elapsed,
            )
            return output

        except Exception as exc:
            elapsed = (time.monotonic() - t_start) * 1000
            logger.exception(
                "Remote tool '%s' on '%s' failed: %s  latency=%.0fms",
                tool_name,
                self._conn.name,
                exc,
                elapsed,
            )
            raise RuntimeError(
                f"Remote tool '{tool_name}' on '{self._conn.name}' failed: {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # health
    # ------------------------------------------------------------------

    async def ping(self) -> bool:
        """Send a ping to check if the connection is alive."""
        if self._session is None:
            return False
        try:
            await self._session.send_ping()
            return True
        except Exception:
            return False
