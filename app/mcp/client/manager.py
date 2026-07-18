"""MCP Connection Manager — manages multiple external MCP server connections.

The manager is the central orchestration point for all remote MCP
servers.  It handles connect, disconnect, tool discovery, health
monitoring, and automatic reconnection.

Design
------
- **Multi-server.**  One manager handles N servers simultaneously.
  Each server gets its own ``MCPClient`` instance.
- **Lazy connection.**  Servers are connected on demand (``connect_all``)
  rather than at import time, giving the application control over
  startup order.
- **Graceful degradation.**  A failed connection to one server does
  not affect others.  Failed servers are flagged ``ERROR`` and can
  be retried.
- **Reconnect.**  ``reconnect()`` tears down and re-establishes a
  connection, useful for recovering from transient failures.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from app.mcp.client.client import MCPClient
from app.mcp.client.models import ConnectionStatus, MCPServerConnection, RemoteToolInfo

logger = logging.getLogger(__name__)


class MCPConnectionManager:
    """Manage connections to multiple external MCP servers.

    Usage::

        manager = MCPConnectionManager()

        conn = MCPServerConnection(
            name="math-server",
            command="python", args=["-m", "math_server"],
        )
        manager.add_server(conn)

        await manager.connect_all()
        await manager.discover_all_tools()
        all_tools = manager.list_all_tools()
        await manager.disconnect_all()
    """

    def __init__(self) -> None:
        self._servers: dict[str, MCPServerConnection] = {}
        self._clients: dict[str, MCPClient] = {}
        self._auto_reconnect: bool = False
        self._heartbeat_interval: float = 30.0

    # ------------------------------------------------------------------
    # configuration
    # ------------------------------------------------------------------

    def configure(
        self,
        *,
        auto_reconnect: bool = False,
        heartbeat_interval: float = 30.0,
    ) -> None:
        """Set operational parameters."""
        self._auto_reconnect = auto_reconnect
        self._heartbeat_interval = heartbeat_interval

    # ------------------------------------------------------------------
    # server management
    # ------------------------------------------------------------------

    def add_server(self, connection: MCPServerConnection) -> None:
        """Register a server without connecting."""
        self._servers[connection.name] = connection
        logger.info("MCP server added: %s", connection.name)

    def remove_server(self, name: str) -> bool:
        """Remove a server (disconnects first if connected)."""
        if name in self._clients:
            try:
                asyncio.run(self._clients[name].disconnect())
            except Exception:
                pass
            del self._clients[name]
        existed = self._servers.pop(name, None) is not None
        if existed:
            logger.info("MCP server removed: %s", name)
        return existed

    # ------------------------------------------------------------------
    # connection lifecycle
    # ------------------------------------------------------------------

    async def connect_all(self) -> dict[str, bool]:
        """Connect to all registered servers.

        Returns
        -------
        dict[str, bool]
            Server name → connection success.
        """
        results: dict[str, bool] = {}
        for name in self._servers:
            try:
                await self.connect_one(name)
                results[name] = True
            except Exception:
                results[name] = False
        return results

    async def connect_one(self, name: str) -> None:
        """Connect to a single server by name."""
        conn = self._servers.get(name)
        if conn is None:
            raise ValueError(f"Unknown server: {name}")

        # Disconnect first if already connected
        if name in self._clients:
            await self._clients[name].disconnect()

        client = MCPClient(conn)
        await client.connect()
        self._clients[name] = client

    async def disconnect_all(self) -> None:
        """Disconnect from all servers."""
        for name in list(self._clients.keys()):
            try:
                await self._clients[name].disconnect()
            except Exception as exc:
                logger.warning("Error disconnecting '%s': %s", name, exc)
        self._clients.clear()

    async def reconnect(self, name: str) -> bool:
        """Reconnect a server (disconnect + connect)."""
        logger.info("Reconnecting to '%s' ...", name)
        t_start = time.monotonic()
        try:
            await self.connect_one(name)
            elapsed = (time.monotonic() - t_start) * 1000
            logger.info("Reconnected to '%s'  latency=%.0fms", name, elapsed)
            return True
        except Exception as exc:
            elapsed = (time.monotonic() - t_start) * 1000
            logger.error(
                "Failed to reconnect to '%s': %s  latency=%.0fms",
                name, exc, elapsed,
            )
            return False

    # ------------------------------------------------------------------
    # tool discovery
    # ------------------------------------------------------------------

    async def discover_all_tools(self) -> dict[str, list[RemoteToolInfo]]:
        """Discover tools on all connected servers."""
        results: dict[str, list[RemoteToolInfo]] = {}
        for name, client in self._clients.items():
            try:
                results[name] = await client.discover_tools()
            except Exception as exc:
                logger.error("Tool discovery failed on '%s': %s", name, exc)
                results[name] = []
        return results

    def list_all_tools(self) -> list[RemoteToolInfo]:
        """Return cached tools from all connected servers."""
        all_tools: list[RemoteToolInfo] = []
        for client in self._clients.values():
            all_tools.extend(client.tools)
        return all_tools

    # ------------------------------------------------------------------
    # health
    # ------------------------------------------------------------------

    async def health_check(self) -> dict[str, bool]:
        """Ping all connected servers."""
        results: dict[str, bool] = {}
        for name, client in self._clients.items():
            results[name] = await client.ping()
        return results

    # ------------------------------------------------------------------
    # queries
    # ------------------------------------------------------------------

    def get_client(self, name: str) -> MCPClient | None:
        """Return the MCP client for a server, or None."""
        return self._clients.get(name)

    def list_servers(self) -> list[dict[str, Any]]:
        """Return metadata for all registered servers."""
        return [
            {
                "name": conn.name,
                "status": conn.status.value,
                "tools_count": len(conn.tools),
                "error": conn.last_error,
            }
            for conn in self._servers.values()
        ]

    @property
    def connected_count(self) -> int:
        return sum(
            1 for c in self._clients.values()
            if c.status == ConnectionStatus.CONNECTED
        )

    @property
    def total_count(self) -> int:
        return len(self._servers)
