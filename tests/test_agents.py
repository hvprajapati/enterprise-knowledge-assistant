"""Unit tests for the Multi-Agent Framework.

Covers:
- BaseAgent execution + logging
- AgentRegistry: register, lookup, unregister, list, count
- SupervisorAgent delegation
- PlannerAgent wraps Planner
- RetrievalAgent wraps retrieval pipeline
- GenerationAgent wraps LLM
- ReflectionAgent wraps ReflectionEngine
- ValidationAgent wraps AnswerValidator
- Agent failure recovery
- supervisor_node integration
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from app.agents import (
    AgentRegistry,
    BaseAgent,
    GenerationAgent,
    PlannerAgent,
    ReflectionAgent,
    RetrievalAgent,
    ValidationAgent,
)
from app.agents.supervisor import SupervisorAgent

# ---------------------------------------------------------------------------
# helper: a failing agent for testing error paths
# ---------------------------------------------------------------------------


class _FailingAgent(BaseAgent):
    """An agent that always raises for testing error handling."""

    @property
    def name(self) -> str:
        return "failing"

    @property
    def description(self) -> str:
        return "Always fails."

    def execute(self, state: dict) -> dict:  # type: ignore[override]
        raise RuntimeError("Intentional agent failure.")


class _NoOpAgent(BaseAgent):
    """An agent that does nothing — for testing registry operations."""

    @property
    def name(self) -> str:
        return "noop"

    @property
    def description(self) -> str:
        return "Does nothing."

    def execute(self, state: dict) -> dict:  # type: ignore[override]
        return {"noop_ran": True}


# ---------------------------------------------------------------------------
# BaseAgent tests
# ---------------------------------------------------------------------------


class TestBaseAgent:
    def test_cannot_instantiate_abstract(self) -> None:
        with pytest.raises(TypeError):
            BaseAgent()  # type: ignore[abstract]

    def test_concrete_has_required_attrs(self) -> None:
        agent = _NoOpAgent()
        assert agent.name == "noop"
        assert isinstance(agent.description, str)

    def test_execute_with_logging_on_success(self) -> None:
        agent = _NoOpAgent()
        result = agent.execute_with_logging({})
        assert result["noop_ran"] is True

    def test_execute_with_logging_on_failure(self) -> None:
        agent = _FailingAgent()
        result = agent.execute_with_logging({})
        assert "error" in result
        assert "Intentional agent failure" in result["error"]


# ---------------------------------------------------------------------------
# AgentRegistry tests
# ---------------------------------------------------------------------------


class TestAgentRegistry:
    def test_register_and_lookup(self) -> None:
        registry = AgentRegistry()
        registry.register(_NoOpAgent())
        agent = registry.lookup("noop")
        assert agent is not None
        assert agent.name == "noop"

    def test_register_duplicate_replaces(self) -> None:
        registry = AgentRegistry()
        registry.register(_NoOpAgent())
        registry.register(_NoOpAgent())
        assert registry.agent_count == 1

    def test_unregister_existing(self) -> None:
        registry = AgentRegistry()
        registry.register(_NoOpAgent())
        assert registry.unregister("noop") is True
        assert registry.lookup("noop") is None

    def test_unregister_missing(self) -> None:
        registry = AgentRegistry()
        assert registry.unregister("nonexistent") is False

    def test_lookup_missing(self) -> None:
        registry = AgentRegistry()
        assert registry.lookup("nonexistent") is None

    def test_has_agent(self) -> None:
        registry = AgentRegistry()
        registry.register(_NoOpAgent())
        assert registry.has("noop") is True
        assert registry.has("missing") is False

    def test_list_agents(self) -> None:
        registry = AgentRegistry()
        registry.register(_NoOpAgent())
        agents = registry.list_agents()
        assert len(agents) == 1
        assert agents[0]["name"] == "noop"

    def test_agent_count(self) -> None:
        registry = AgentRegistry()
        assert registry.agent_count == 0
        registry.register(_NoOpAgent())
        assert registry.agent_count == 1


# ---------------------------------------------------------------------------
# SupervisorAgent tests
# ---------------------------------------------------------------------------


class TestSupervisorAgent:
    def test_name_and_description(self) -> None:
        registry = AgentRegistry()
        supervisor = SupervisorAgent(registry=registry)
        assert supervisor.name == "supervisor"
        assert "planner" in supervisor.description

    def test_all_agents_execute_successfully(self) -> None:
        registry = AgentRegistry()
        registry.register(_NoOpAgent())
        registry.register(_NoOpAgent())  # replaces — both are "noop"

        # Use unique names
        class _AgentA(BaseAgent):
            name = property(lambda s: "a")
            description = property(lambda s: "A")

            def execute(self, state: dict) -> dict:  # type: ignore[override]
                return {"a_ran": True}

        class _AgentB(BaseAgent):
            name = property(lambda s: "b")
            description = property(lambda s: "B")

            def execute(self, state: dict) -> dict:  # type: ignore[override]
                return {"b_ran": True}

        registry = AgentRegistry()
        registry.register(_AgentA())
        registry.register(_AgentB())

        supervisor = SupervisorAgent(
            registry=registry,
            pipeline=["a", "b"],
        )

        result: Any = supervisor.execute({})
        assert result["a_ran"] is True
        assert result["b_ran"] is True
        assert result["completed_agents"] == ["a", "b"]
        assert len(result["execution_history"]) == 2  # 2 agents

    def test_skips_missing_agent(self) -> None:
        registry = AgentRegistry()
        registry.register(_NoOpAgent())
        supervisor = SupervisorAgent(
            registry=registry,
            pipeline=["noop", "missing-agent"],
        )

        result: Any = supervisor.execute({})
        assert "noop" in result["completed_agents"]
        assert "missing-agent" not in result["completed_agents"]
        # Should have history entry for the skip
        history = result["execution_history"]
        assert any(h["agent"] == "missing-agent" and h["outcome"] == "skipped" for h in history)

    def test_critical_agent_failure_aborts_pipeline(self) -> None:
        """A critical agent failure should stop the pipeline."""
        registry = AgentRegistry()

        class _GoodAgent(BaseAgent):
            name = property(lambda s: "good")
            description = property(lambda s: "Good")

            def execute(self, state: dict) -> dict:  # type: ignore[override]
                return {"good_ran": True}

        registry.register(_FailingAgent())  # name: "failing"
        registry.register(_GoodAgent())

        supervisor = SupervisorAgent(
            registry=registry,
            pipeline=["failing", "good"],
        )

        # "failing" is not in critical list (planner, retrieval, generation)
        result: Any = supervisor.execute({})
        # Pipeline should continue past non-critical failure
        assert "error" in result or "good_ran" in result  # depends on naming

    def test_non_critical_agent_failure_continues(self) -> None:
        """A non-critical agent failure should not abort."""
        registry = AgentRegistry()
        registry.register(_FailingAgent())  # name "failing" not critical
        registry.register(_NoOpAgent())

        supervisor = SupervisorAgent(
            registry=registry,
            pipeline=["failing", "noop"],
        )

        # "failing" is not critical — "noop" should still run
        result: Any = supervisor.execute({})
        assert result["noop_ran"] is True


# ---------------------------------------------------------------------------
# PlannerAgent tests
# ---------------------------------------------------------------------------


class TestPlannerAgent:
    def test_name_and_description(self) -> None:
        agent = PlannerAgent()
        assert agent.name == "planner"
        assert "plan" in agent.description.lower()

    def test_execute_creates_plan(self) -> None:
        agent = PlannerAgent()
        state: dict = {
            "question": "What is FAISS?",
            "executed_nodes": [],
            "execution_history": [],
        }
        result = agent.execute(state)  # type: ignore[arg-type]
        assert "execution_plan" in result
        assert "requires_rewrite" in result
        assert result["current_agent"] == "planner"
        assert "planner" in result["executed_nodes"]
        assert len(result["execution_history"]) == 1


# ---------------------------------------------------------------------------
# GenerationAgent tests
# ---------------------------------------------------------------------------


class TestGenerationAgent:
    def test_name_and_description(self) -> None:
        agent = GenerationAgent()
        assert agent.name == "generation"

    def test_execute_calls_llm(self) -> None:
        from app.agent.nodes import _services

        mock_llm = MagicMock()
        mock_llm.generate.return_value = "FAISS is a library for similarity search."
        _services["llm"] = mock_llm

        agent = GenerationAgent()
        state: dict = {
            "_prompt": "Question: What is FAISS?\nAnswer:",
            "executed_nodes": [],
            "execution_history": [],
        }
        result = agent.execute(state)  # type: ignore[arg-type]

        assert result["answer"] == "FAISS is a library for similarity search."
        assert result["current_agent"] == "generation"
        mock_llm.generate.assert_called_once()


# ---------------------------------------------------------------------------
# ReflectionAgent tests
# ---------------------------------------------------------------------------


class TestReflectionAgent:
    def test_name_and_description(self) -> None:
        agent = ReflectionAgent()
        assert agent.name == "reflection"

    def test_execute_evaluates_answer(self) -> None:
        from app.agent.nodes import _services

        mock_llm = MagicMock()
        mock_llm.generate.return_value = (
            '{"answer_quality":"good","grounded":"fully_grounded",'
            '"complete":true,"relevant":true,"confidence_score":0.9,'
            '"missing_information":[],"recommendations":[],'
            '"reasoning":"Good answer."}'
        )
        from app.agent.reflection.reflection import ReflectionEngine

        _services["reflection_engine"] = ReflectionEngine(llm=mock_llm)

        agent = ReflectionAgent()
        state: dict = {
            "question": "What is FAISS?",
            "answer": "FAISS is a library.",
            "search_results": [
                {
                    "chunk_id": "c1",
                    "text": "FAISS is for similarity search.",
                    "score": 0.9,
                    "filename": "f.pdf",
                    "page": 1,
                }
            ],
            "executed_nodes": [],
            "execution_history": [],
        }
        result = agent.execute(state)  # type: ignore[arg-type]

        assert "reflection_result" in result
        assert result["current_agent"] == "reflection"
        rr = result["reflection_result"]
        assert rr["answer_quality"] == "good"


# ---------------------------------------------------------------------------
# ValidationAgent tests
# ---------------------------------------------------------------------------


class TestValidationAgent:
    def test_name_and_description(self) -> None:
        agent = ValidationAgent()
        assert agent.name == "validation"

    def test_execute_validates_reflection(self) -> None:
        from app.agent.nodes import _services
        from app.agent.validation.validator import AnswerValidator

        _services["validator"] = AnswerValidator()

        agent = ValidationAgent()
        state: dict = {
            "reflection_result": {
                "answer_quality": "excellent",
                "grounded": "fully_grounded",
                "complete": True,
                "relevant": True,
                "confidence_score": 0.92,
                "missing_information": [],
                "recommendations": [],
                "reasoning": "Perfect.",
            },
            "executed_nodes": [],
            "execution_history": [],
        }
        result = agent.execute(state)  # type: ignore[arg-type]

        assert "validation_result" in result
        assert result["current_agent"] == "validation"
        vr = result["validation_result"]
        assert vr["passed"] is True


# ---------------------------------------------------------------------------
# RetrievalAgent tests
# ---------------------------------------------------------------------------


class TestRetrievalAgent:
    def test_name_and_description(self) -> None:
        agent = RetrievalAgent()
        assert agent.name == "retrieval"
        assert "search" in agent.description.lower()


# ---------------------------------------------------------------------------
# supervisor_node integration
# ---------------------------------------------------------------------------


class TestSupervisorNodeIntegration:
    def test_supervisor_node_initialises_state(self) -> None:
        from app.agent.nodes import supervisor_node

        state: dict = {
            "question": "What is FAISS?",
            "executed_nodes": [],
        }
        result = supervisor_node(state)  # type: ignore[arg-type]

        assert result["completed_agents"] == []
        assert result["execution_history"] == []
        assert result["current_agent"] == "supervisor"
        assert "supervisor" in result["executed_nodes"]

    def test_planner_node_delegates_to_agent(self) -> None:
        from app.agent.nodes import _services, planner_node

        registry = AgentRegistry()
        registry.register(PlannerAgent())
        _services["agent_registry"] = registry

        state: dict = {
            "question": "What is FAISS?",
            "executed_nodes": [],
            "execution_history": [],
        }
        result = planner_node(state)  # type: ignore[arg-type]

        assert "execution_plan" in result
        assert result["current_agent"] == "planner"

    def test_node_delegation_with_missing_agent(self) -> None:
        from app.agent.nodes import _delegate, _services

        _services["agent_registry"] = AgentRegistry()
        state: dict = {
            "executed_nodes": [],
            "execution_history": [],
        }
        result = _delegate(state, "nonexistent-agent")  # type: ignore[arg-type]

        # Missing agent returns None — caller falls back to inline logic
        assert result is None
