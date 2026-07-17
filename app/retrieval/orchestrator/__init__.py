"""Retrieval Orchestrator — intelligent pipeline planning.

Decides which retrieval stages should execute based on the question
type, minimising latency and LLM cost while maintaining answer quality.
"""

from app.retrieval.orchestrator.models import QuestionType, RetrievalPlan
from app.retrieval.orchestrator.orchestrator import RetrievalOrchestrator

__all__ = [
    "QuestionType",
    "RetrievalOrchestrator",
    "RetrievalPlan",
]
