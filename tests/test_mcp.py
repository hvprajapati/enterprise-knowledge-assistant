"""Unit tests for the MCP Server.

Covers:
- MCPToolAdapter: to_mcp_tool, invoke success, invoke failure
- MCPRegistry: sync, list_tools, lookup, register, unregister
- MCPServerConfig: defaults, custom values
- MCPServer: creation, handler registration
- Integration: adapter wrapping existing tools
"""

from __future__ import annotations

import asyncio
import json

import pytest

from app.mcp import MCPRegistry, MCPServer, MCPServerConfig, MCPToolAdapter
from app.tools import BaseTool, ToolRegistry
from app.tools.calculator import CalculatorTool
from app.tools.current_time import CurrentTimeTool

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


class _MockTool(BaseTool):
    """A simple mock tool for testing adapter and registry behavior."""

    @property
    def name(self) -> str:
        return "mock-tool"

    @property
    def description(self) -> str:
        return "A mock tool for testing."

    @property
    def schema(self) -> dict:
        return {
            "type": "object",
            "properties": {"value": {"type": "string"}},
            "required": ["value"],
        }

    def invoke(self, **kwargs: object) -> dict[str, object]:
        return {"result": f"got: {kwargs.get('value', '')}"}


class _ErrorTool(BaseTool):
    """A tool that always raises for testing error paths."""

    @property
    def name(self) -> str:
        return "error-tool"

    @property
    def description(self) -> str:
        return "Always fails."

    @property
    def schema(self) -> dict:
        return {"type": "object", "properties": {}, "required": []}

    def invoke(self, **kwargs: object) -> dict[str, object]:
        raise RuntimeError("Intentional tool failure.")


# ---------------------------------------------------------------------------
# MCPServerConfig tests
# ---------------------------------------------------------------------------


class TestMCPServerConfig:
    def test_defaults(self) -> None:
        config = MCPServerConfig()
        assert config.name == "Enterprise Knowledge Assistant"
        assert config.version == "0.1.0"
        assert config.enable is True

    def test_custom_values(self) -> None:
        config = MCPServerConfig(name="Test Server", version="2.0.0", enable=False)
        assert config.name == "Test Server"
        assert config.version == "2.0.0"
        assert config.enable is False


# ---------------------------------------------------------------------------
# MCPToolAdapter tests
# ---------------------------------------------------------------------------


class TestMCPToolAdapter:
    def test_name_description_schema(self) -> None:
        adapter = MCPToolAdapter(CalculatorTool())
        assert adapter.name == "calculator"
        assert "math" in adapter.description.lower()
        assert "expression" in adapter.schema.get("required", [])

    def test_to_mcp_tool(self) -> None:
        adapter = MCPToolAdapter(CalculatorTool())
        mcp_tool = adapter.to_mcp_tool()
        assert mcp_tool.name == "calculator"
        assert mcp_tool.description is not None
        assert "expression" in mcp_tool.inputSchema.get("required", [])

    def test_invoke_success(self) -> None:
        adapter = MCPToolAdapter(_MockTool())
        result = adapter.invoke({"value": "hello"})
        assert len(result) == 1
        assert result[0].type == "text"
        data = json.loads(result[0].text)
        assert data["result"] == "got: hello"

    def test_invoke_calculator(self) -> None:
        adapter = MCPToolAdapter(CalculatorTool())
        result = adapter.invoke({"expression": "2 + 3"})
        data = json.loads(result[0].text)
        assert data["result"] == 5

    def test_invoke_current_time(self) -> None:
        adapter = MCPToolAdapter(CurrentTimeTool())
        result = adapter.invoke({})
        data = json.loads(result[0].text)
        assert "result" in data
        assert "timezone" in data

    def test_invoke_failure(self) -> None:
        adapter = MCPToolAdapter(_ErrorTool())
        result = adapter.invoke({})
        data = json.loads(result[0].text)
        assert "error" in data
        assert "Intentional tool failure" in data["error"]

    def test_invoke_invalid_arguments(self) -> None:
        adapter = MCPToolAdapter(CalculatorTool())
        result = adapter.invoke({"expression": ""})
        data = json.loads(result[0].text)
        assert "error" in data


# ---------------------------------------------------------------------------
# MCPRegistry tests
# ---------------------------------------------------------------------------


class TestMCPRegistry:
    def test_sync_creates_adapters(self) -> None:
        tool_registry = ToolRegistry()
        tool_registry.register(CalculatorTool())
        tool_registry.register(CurrentTimeTool())

        mcp_registry = MCPRegistry(tool_registry=tool_registry)
        mcp_registry.sync()

        assert mcp_registry.tool_count == 2
        assert mcp_registry.has("calculator")
        assert mcp_registry.has("current-time")

    def test_list_tools_returns_mcp_tools(self) -> None:
        tool_registry = ToolRegistry()
        tool_registry.register(CalculatorTool())
        mcp_registry = MCPRegistry(tool_registry=tool_registry)
        mcp_registry.sync()

        tools = mcp_registry.list_tools()
        assert len(tools) == 1
        assert tools[0].name == "calculator"

    def test_lookup(self) -> None:
        tool_registry = ToolRegistry()
        tool_registry.register(CalculatorTool())
        mcp_registry = MCPRegistry(tool_registry=tool_registry)
        mcp_registry.sync()

        adapter = mcp_registry.lookup("calculator")
        assert adapter is not None
        assert adapter.name == "calculator"

    def test_lookup_missing(self) -> None:
        tool_registry = ToolRegistry()
        mcp_registry = MCPRegistry(tool_registry=tool_registry)
        assert mcp_registry.lookup("nonexistent") is None

    def test_register_tool(self) -> None:
        tool_registry = ToolRegistry()
        mcp_registry = MCPRegistry(tool_registry=tool_registry)
        mcp_registry.register_tool(CalculatorTool())

        assert mcp_registry.tool_count == 1
        assert mcp_registry.has("calculator")
        # Also in the underlying ToolRegistry
        assert tool_registry.has("calculator")

    def test_unregister_tool(self) -> None:
        tool_registry = ToolRegistry()
        mcp_registry = MCPRegistry(tool_registry=tool_registry)
        mcp_registry.register_tool(CalculatorTool())
        mcp_registry.register_tool(CurrentTimeTool())

        removed = mcp_registry.unregister_tool("calculator")
        assert removed is True
        assert mcp_registry.tool_count == 1
        assert not mcp_registry.has("calculator")

    def test_unregister_missing(self) -> None:
        mcp_registry = MCPRegistry(tool_registry=ToolRegistry())
        assert mcp_registry.unregister_tool("nonexistent") is False


# ---------------------------------------------------------------------------
# MCPServer tests
# ---------------------------------------------------------------------------


class TestMCPServer:
    def test_creation_with_registry(self) -> None:
        tool_registry = ToolRegistry()
        mcp_registry = MCPRegistry(tool_registry=tool_registry)
        server = MCPServer(mcp_registry=mcp_registry)
        assert server is not None

    def test_creation_with_config(self) -> None:
        tool_registry = ToolRegistry()
        mcp_registry = MCPRegistry(tool_registry=tool_registry)
        config = MCPServerConfig(name="Test", version="1.0")
        server = MCPServer(mcp_registry=mcp_registry, config=config)
        assert server is not None

    def test_server_disabled_does_not_run(self) -> None:
        tool_registry = ToolRegistry()
        mcp_registry = MCPRegistry(tool_registry=tool_registry)
        config = MCPServerConfig(enable=False)
        server = MCPServer(mcp_registry=mcp_registry, config=config)

        # serve() should return immediately without error
        async def _run() -> None:
            await server.serve()

        asyncio.run(_run())  # should complete without blocking

    def test_create_mcp_server_factory(self) -> None:
        from app.mcp.server import create_mcp_server

        tool_registry = ToolRegistry()
        mcp_registry = MCPRegistry(tool_registry=tool_registry)
        config = MCPServerConfig(name="Factory Test")

        server = create_mcp_server(mcp_registry=mcp_registry, config=config)
        assert server is not None


# ---------------------------------------------------------------------------
# Integration: full adapter lifecycle
# ---------------------------------------------------------------------------


class TestMCPIntegration:
    def test_full_tool_lifecycle(self) -> None:
        """Test: register → sync → lookup → invoke."""
        tool_registry = ToolRegistry()
        tool_registry.register(_MockTool())

        mcp_registry = MCPRegistry(tool_registry=tool_registry)
        mcp_registry.sync()

        # Discovery
        tools = mcp_registry.list_tools()
        assert len(tools) == 1
        assert tools[0].name == "mock-tool"

        # Lookup
        adapter = mcp_registry.lookup("mock-tool")
        assert adapter is not None

    def test_adapter_invoke_through_registry(self) -> None:
        tool_registry = ToolRegistry()
        tool_registry.register(_MockTool())
        mcp_registry = MCPRegistry(tool_registry=tool_registry)
        mcp_registry.sync()

        adapter = mcp_registry.lookup("mock-tool")
        assert adapter is not None

        result = adapter.invoke({"value": "test"})
        data = json.loads(result[0].text)
        assert data["result"] == "got: test"
