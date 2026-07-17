"""Multi-Agent Framework for the Enterprise Knowledge Assistant.

Each agent has a **single responsibility** and communicates through
shared ``AgentState``.  The ``SupervisorAgent`` coordinates execution
order and handles failures.

Components
----------
- ``BaseAgent`` — abstract interface for all agents.
- ``AgentRegistry`` — catalogue of available agents.
- ``SupervisorAgent`` — coordinates the agent pipeline.
- ``PlannerAgent`` — question analysis and execution planning.
- ``RetrievalAgent`` — FAISS search, reranking, prompt building.
- ``GenerationAgent`` — LLM answer generation.
- ``ReflectionAgent`` — answer quality evaluation.
- ``ValidationAgent`` — threshold-based quality gating.

Adding a new agent
------------------
1. Subclass ``BaseAgent`` and implement ``name``, ``description``,
   ``execute(state)``.
2. Register it: ``registry.register(MyAgent())``.
3. Add its name to the supervisor's pipeline.

No graph changes needed.
"""

from __future__ import annotations

from app.agents.base import BaseAgent
from app.agents.generation_agent import GenerationAgent
from app.agents.planner_agent import PlannerAgent
from app.agents.reflection_agent import ReflectionAgent
from app.agents.registry import AgentRegistry
from app.agents.retrieval_agent import RetrievalAgent
from app.agents.supervisor import SupervisorAgent
from app.agents.validation_agent import ValidationAgent

__all__ = [
    "AgentRegistry",
    "BaseAgent",
    "GenerationAgent",
    "PlannerAgent",
    "ReflectionAgent",
    "RetrievalAgent",
    "SupervisorAgent",
    "ValidationAgent",
]
