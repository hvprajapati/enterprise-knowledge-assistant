"""Unit tests for the Tool Calling Framework.

Covers:
- BaseTool abstract contract
- CalculatorTool: basic arithmetic, math functions, invalid expressions
- CurrentTimeTool: returns expected keys
- DocumentSearchTool: unconfigured error, configured search
- ToolRegistry: register, unregister, lookup, list_tools, has, duplicate
- ToolExecutor: success, missing tool, invalid args, tool failure
- ToolResult / ToolDecision model validation
- tool_node integration
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from app.tools import BaseTool, ToolDecision, ToolExecutor, ToolRegistry, ToolResult
from app.tools.calculator import CalculatorTool
from app.tools.current_time import CurrentTimeTool
from app.tools.document_search import DocumentSearchTool

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


class _FailingTool(BaseTool):
    """A tool that always raises for testing error paths."""

    @property
    def name(self) -> str:
        return "failing-tool"

    @property
    def description(self) -> str:
        return "Always fails."

    @property
    def schema(self) -> dict[str, object]:
        return {"type": "object", "properties": {}, "required": []}

    def invoke(self, **kwargs: object) -> dict[str, object]:
        raise RuntimeError("Intentional failure for testing.")


class _SchemaTool(BaseTool):
    """A tool with required arguments for validation testing."""

    @property
    def name(self) -> str:
        return "schema-tool"

    @property
    def description(self) -> str:
        return "Requires arguments."

    @property
    def schema(self) -> dict[str, object]:
        return {
            "type": "object",
            "properties": {
                "text": {"type": "string"},
                "count": {"type": "integer"},
            },
            "required": ["text", "count"],
        }

    def invoke(self, **kwargs: object) -> dict[str, object]:
        return {"result": f"{kwargs['text']} × {kwargs['count']}"}


# ---------------------------------------------------------------------------
# BaseTool tests
# ---------------------------------------------------------------------------


class TestBaseTool:
    def test_cannot_instantiate_abstract(self) -> None:
        with pytest.raises(TypeError):
            BaseTool()  # type: ignore[abstract]

    def test_concrete_tool_has_required_attrs(self) -> None:
        tool = CalculatorTool()
        assert tool.name == "calculator"
        assert isinstance(tool.description, str) and len(tool.description) > 0
        assert "expression" in tool.schema.get("required", [])


# ---------------------------------------------------------------------------
# CalculatorTool tests
# ---------------------------------------------------------------------------


class TestCalculatorTool:
    def test_simple_arithmetic(self) -> None:
        tool = CalculatorTool()
        result = tool.invoke(expression="2 + 2")
        assert result["result"] == 4
        assert result["expression"] == "2 + 2"

    def test_complex_expression(self) -> None:
        tool = CalculatorTool()
        result = tool.invoke(expression="(10 + 5) * 3 / 2")
        assert result["result"] == 22.5

    def test_math_function(self) -> None:
        tool = CalculatorTool()
        result = tool.invoke(expression="sqrt(16) + abs(-5)")
        assert result["result"] == 9.0  # 4 + 5

    def test_power_operator(self) -> None:
        tool = CalculatorTool()
        result = tool.invoke(expression="2 ** 10")
        assert result["result"] == 1024

    def test_empty_expression_raises(self) -> None:
        tool = CalculatorTool()
        with pytest.raises(ValueError, match="empty"):
            tool.invoke(expression="")

    def test_invalid_syntax_raises(self) -> None:
        tool = CalculatorTool()
        with pytest.raises(ValueError, match="Invalid expression syntax"):
            tool.invoke(expression="2 +* 3")

    def test_unknown_function_raises(self) -> None:
        tool = CalculatorTool()
        with pytest.raises(ValueError, match="Unknown name"):
            tool.invoke(expression="hack_the_planet()")


# ---------------------------------------------------------------------------
# CurrentTimeTool tests
# ---------------------------------------------------------------------------


class TestCurrentTimeTool:
    def test_returns_expected_keys(self) -> None:
        tool = CurrentTimeTool()
        result = tool.invoke()
        assert "result" in result
        assert "timestamp_unix" in result
        assert "day_of_week" in result
        assert "date" in result
        assert "time" in result
        assert result["timezone"] == "UTC"

    def test_result_is_iso_format(self) -> None:
        tool = CurrentTimeTool()
        result = tool.invoke()
        import datetime
        # Should parse without error
        datetime.datetime.fromisoformat(str(result["result"]))

    def test_no_args_needed(self) -> None:
        tool = CurrentTimeTool()
        assert tool.schema["required"] == []
        result = tool.invoke()
        assert result["result"]  # not empty


# ---------------------------------------------------------------------------
# DocumentSearchTool tests
# ---------------------------------------------------------------------------


class TestDocumentSearchTool:
    def test_unconfigured_raises(self) -> None:
        tool = DocumentSearchTool()
        with pytest.raises(RuntimeError, match="not configured"):
            tool.invoke(query="test")

    def test_empty_query_raises_when_configured(self) -> None:
        tool = DocumentSearchTool()
        mock_qs = MagicMock()
        tool.configure(mock_qs)
        with pytest.raises(ValueError, match="empty"):
            tool.invoke(query="   ")

    def test_configured_search_returns_result(self) -> None:
        tool = DocumentSearchTool()
        mock_qs = MagicMock()
        mock_qs.answer.return_value = {
            "answer": "FAISS is a library for similarity search.",
            "sources": [
                {"text": "FAISS overview", "filename": "faiss.pdf", "score": 0.95},
                {"text": "GPU acceleration", "filename": "faiss.pdf", "score": 0.87},
            ],
        }
        tool.configure(mock_qs)

        result = tool.invoke(query="What is FAISS?", top_k=2)

        assert "result" in result
        assert len(result["passages"]) == 2
        assert result["total_sources"] == 2
        mock_qs.answer.assert_called_once_with("What is FAISS?")


# ---------------------------------------------------------------------------
# ToolRegistry tests
# ---------------------------------------------------------------------------


class TestToolRegistry:
    def test_register_and_lookup(self) -> None:
        registry = ToolRegistry()
        registry.register(CalculatorTool())
        tool = registry.lookup("calculator")
        assert tool is not None
        assert tool.name == "calculator"

    def test_register_duplicate_replaces(self) -> None:
        registry = ToolRegistry()
        registry.register(CalculatorTool())
        registry.register(CalculatorTool())  # should warn but not crash
        assert registry.tool_count == 1

    def test_unregister_existing(self) -> None:
        registry = ToolRegistry()
        registry.register(CalculatorTool())
        assert registry.unregister("calculator") is True
        assert registry.lookup("calculator") is None
        assert registry.tool_count == 0

    def test_unregister_missing(self) -> None:
        registry = ToolRegistry()
        assert registry.unregister("nonexistent") is False

    def test_lookup_missing_returns_none(self) -> None:
        registry = ToolRegistry()
        assert registry.lookup("nonexistent") is None

    def test_has_tool(self) -> None:
        registry = ToolRegistry()
        registry.register(CalculatorTool())
        assert registry.has("calculator") is True
        assert registry.has("nonexistent") is False

    def test_list_tools_returns_metadata(self) -> None:
        registry = ToolRegistry()
        registry.register(CalculatorTool())
        registry.register(CurrentTimeTool())

        tools = registry.list_tools()
        assert len(tools) == 2
        names = {t["name"] for t in tools}
        assert names == {"calculator", "current-time"}
        for t in tools:
            assert "description" in t
            assert "schema" in t

    def test_tool_count(self) -> None:
        registry = ToolRegistry()
        assert registry.tool_count == 0
        registry.register(CalculatorTool())
        assert registry.tool_count == 1
        registry.register(CurrentTimeTool())
        assert registry.tool_count == 2


# ---------------------------------------------------------------------------
# ToolExecutor tests
# ---------------------------------------------------------------------------


class TestToolExecutor:
    def test_execute_success(self) -> None:
        registry = ToolRegistry()
        registry.register(CalculatorTool())
        executor = ToolExecutor(registry=registry)

        result = executor.execute("calculator", {"expression": "2 + 3"})
        assert result.success is True
        assert result.tool_name == "calculator"
        assert result.output["result"] == 5
        assert result.execution_time_ms >= 0

    def test_execute_missing_tool(self) -> None:
        registry = ToolRegistry()
        executor = ToolExecutor(registry=registry)

        result = executor.execute("nonexistent", {})
        assert result.success is False
        assert "not registered" in result.error
        assert result.execution_time_ms >= 0

    def test_execute_invalid_arguments_missing_required(self) -> None:
        registry = ToolRegistry()
        registry.register(_SchemaTool())
        executor = ToolExecutor(registry=registry)

        result = executor.execute("schema-tool", {})
        assert result.success is False
        assert "Missing required arguments" in result.error
        assert "text" in result.error

    def test_execute_invalid_arguments_wrong_type(self) -> None:
        registry = ToolRegistry()
        registry.register(_SchemaTool())
        executor = ToolExecutor(registry=registry)

        result = executor.execute("schema-tool", {"text": 123, "count": "not_int"})
        assert result.success is False
        assert "type mismatch" in result.error.lower() or "Type mismatch" in result.error

    def test_execute_tool_raises_exception(self) -> None:
        registry = ToolRegistry()
        registry.register(_FailingTool())
        executor = ToolExecutor(registry=registry)

        result = executor.execute("failing-tool", {})
        assert result.success is False
        assert "Intentional failure" in result.error
        assert result.tool_name == "failing-tool"

    def test_execute_with_empty_arguments(self) -> None:
        registry = ToolRegistry()
        registry.register(CurrentTimeTool())
        executor = ToolExecutor(registry=registry)

        result = executor.execute("current-time")
        assert result.success is True
        assert "result" in result.output

    def test_tool_result_from_success_factory(self) -> None:
        result = ToolResult.from_success(
            tool_name="test-tool",
            output={"result": 42},
            execution_time_ms=15.5,
        )
        assert result.success is True
        assert result.output["result"] == 42
        assert result.execution_time_ms == 15.5

    def test_tool_result_from_error_factory(self) -> None:
        result = ToolResult.from_error(
            tool_name="test-tool",
            error="Something went wrong.",
            execution_time_ms=3.2,
        )
        assert result.success is False
        assert result.error == "Something went wrong."
        assert result.output == {}

    def test_tool_result_format_for_prompt_success(self) -> None:
        result = ToolResult.from_success(
            tool_name="calculator",
            output={"result": 42, "expression": "40 + 2"},
        )
        formatted = result.format_for_prompt()
        assert "[Tool: calculator]" in formatted
        assert "result: 42" in formatted
        assert "expression: 40 + 2" in formatted

    def test_tool_result_format_for_prompt_error(self) -> None:
        result = ToolResult.from_error(
            tool_name="calculator",
            error="Division by zero",
        )
        formatted = result.format_for_prompt()
        assert "[Tool: calculator]" in formatted
        assert "ERROR: Division by zero" in formatted


# ---------------------------------------------------------------------------
# ToolDecision tests
# ---------------------------------------------------------------------------


class TestToolDecision:
    def test_skip_tools_factory(self) -> None:
        decision = ToolDecision.skip_tools()
        assert decision.use_tool is False
        assert decision.tool_name == ""
        assert decision.arguments == {}
        assert decision.confidence == 1.0

    def test_full_decision(self) -> None:
        decision = ToolDecision(
            use_tool=True,
            tool_name="calculator",
            arguments={"expression": "2 + 2"},
            confidence=0.95,
            reasoning="User asked for a calculation.",
        )
        assert decision.use_tool is True
        assert decision.tool_name == "calculator"

    def test_to_log_dict(self) -> None:
        decision = ToolDecision(
            use_tool=True,
            tool_name="calculator",
            arguments={"expression": "2 + 2"},
            reasoning="Math question detected.",
        )
        log = decision.to_log_dict()
        assert log["use_tool"] is True
        assert log["tool_name"] == "calculator"
        assert "Math question" in log["reasoning"]

    def test_model_dump_for_graph(self) -> None:
        decision = ToolDecision(
            use_tool=False,
            tool_name="",
            reasoning="No tool needed.",
        )
        dumped = decision.model_dump()
        assert dumped["use_tool"] is False
        assert dumped["tool_name"] == ""


# ---------------------------------------------------------------------------
# integration-style: tool node
# ---------------------------------------------------------------------------


class TestToolNodeIntegration:
    """Lightweight tests that exercise the tool_node function directly."""

    def test_tool_node_skip_when_no_tools_expected(self) -> None:
        from app.agent.nodes import tool_node

        state: dict = {
            "question": "What is FAISS?",
            "execution_plan": {"expected_tools": []},
            "executed_nodes": ["planner"],
        }

        result = tool_node(state)  # type: ignore[arg-type]

        assert "tool_decision" in result
        assert result["tool_decision"]["use_tool"] is False
        assert result["tool_result"] == {}
        assert "tool" in result["executed_nodes"]

    def test_tool_node_executes_calculator(self) -> None:
        from app.agent.nodes import _services, tool_node

        registry = ToolRegistry()
        registry.register(CalculatorTool())
        _services["tool_executor"] = ToolExecutor(registry=registry)

        state: dict = {
            "question": "10 + 5",
            "execution_plan": {"expected_tools": ["calculator"]},
            "executed_nodes": ["planner"],
        }

        result = tool_node(state)  # type: ignore[arg-type]

        assert result["tool_decision"]["use_tool"] is True
        assert result["tool_decision"]["tool_name"] == "calculator"
        tool_result = result["tool_result"]
        assert tool_result["success"] is True
        assert tool_result["tool_name"] == "calculator"

    def test_tool_node_handles_missing_tool_gracefully(self) -> None:
        from app.agent.nodes import _services, tool_node

        _services["tool_executor"] = ToolExecutor(registry=ToolRegistry())

        state: dict = {
            "question": "Use a missing tool.",
            "execution_plan": {"expected_tools": ["nonexistent-tool"]},
            "executed_nodes": ["planner"],
        }

        result = tool_node(state)  # type: ignore[arg-type]

        assert result["tool_decision"]["use_tool"] is True
        tool_result = result["tool_result"]
        assert tool_result["success"] is False
        assert "not registered" in tool_result["error"]
        # Graph does NOT crash
        assert "tool" in result["executed_nodes"]

    def test_tool_node_engine_not_configured(self) -> None:
        from app.agent.nodes import _services, tool_node

        _services.pop("tool_executor", None)

        state: dict = {
            "question": "Q",
            "execution_plan": {"expected_tools": ["calculator"]},
            "executed_nodes": ["planner"],
        }

        with pytest.raises(RuntimeError, match="tool_executor"):
            tool_node(state)  # type: ignore[arg-type]
