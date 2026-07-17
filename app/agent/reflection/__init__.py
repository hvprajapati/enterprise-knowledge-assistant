"""Agent Reflection — answer evaluation without modification.

The reflection sub-package evaluates generated answers for quality,
groundedness, completeness, and relevance.  It is a **read-only**
inspection layer — it never changes the answer.

Components
----------
- ``ReflectionResult`` — structured Pydantic model for evaluation output.
- ``ReflectionEngine`` — LLM-powered evaluator with graceful degradation.
- ``reflection_node`` — LangGraph node (in ``app.agent.nodes``).
"""

from __future__ import annotations

from app.agent.reflection.models import (
    AnswerQuality,
    GroundedStatus,
    ReflectionResult,
)
from app.agent.reflection.reflection import ReflectionEngine

__all__ = [
    "AnswerQuality",
    "GroundedStatus",
    "ReflectionEngine",
    "ReflectionResult",
]
