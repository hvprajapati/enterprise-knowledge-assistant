"""Agent Planner — analyses questions and creates ``ExecutionPlan``.

Currently rule-based.  The ``Planner`` interface is deliberately
narrow so an LLM-based planner can replace it later without touching
any downstream node.
"""

from __future__ import annotations

import logging
import time

from app.agent.planner.classifier import classify
from app.agent.planner.models import (
    ExecutionPlan,
    QuestionType,
    RetrievalStrategy,
)

logger = logging.getLogger(__name__)

# Mapping: QuestionType → (RetrievalStrategy, config flags)
_STRATEGY_MAP: dict[QuestionType, dict[str, object]] = {
    QuestionType.FACTUAL: {
        "retrieval_strategy": RetrievalStrategy.DENSE_ONLY,
        "requires_rewrite": False,
        "requires_self_query": False,
        "requires_multi_query": False,
        "requires_parent_retrieval": False,
        "requires_context_compression": False,
    },
    QuestionType.COMPARISON: {
        "retrieval_strategy": RetrievalStrategy.FULL_PIPELINE,
        "requires_rewrite": True,
        "requires_self_query": False,
        "requires_multi_query": True,
        "requires_parent_retrieval": True,
        "requires_context_compression": True,
    },
    QuestionType.SUMMARIZATION: {
        "retrieval_strategy": RetrievalStrategy.HYBRID,
        "requires_rewrite": True,
        "requires_self_query": False,
        "requires_multi_query": False,
        "requires_parent_retrieval": True,
        "requires_context_compression": True,
    },
    QuestionType.TROUBLESHOOTING: {
        "retrieval_strategy": RetrievalStrategy.FULL_PIPELINE,
        "requires_rewrite": True,
        "requires_self_query": False,
        "requires_multi_query": True,
        "requires_parent_retrieval": True,
        "requires_context_compression": True,
    },
    QuestionType.METADATA_LOOKUP: {
        "retrieval_strategy": RetrievalStrategy.METADATA_FILTERED,
        "requires_rewrite": False,
        "requires_self_query": True,
        "requires_multi_query": False,
        "requires_parent_retrieval": False,
        "requires_context_compression": False,
    },
    QuestionType.BROAD_RESEARCH: {
        "retrieval_strategy": RetrievalStrategy.MULTI_QUERY,
        "requires_rewrite": True,
        "requires_self_query": False,
        "requires_multi_query": True,
        "requires_parent_retrieval": False,
        "requires_context_compression": True,
    },
    QuestionType.CONVERSATIONAL: {
        "retrieval_strategy": RetrievalStrategy.DENSE_ONLY,
        "requires_rewrite": False,
        "requires_self_query": False,
        "requires_multi_query": False,
        "requires_parent_retrieval": False,
        "requires_context_compression": False,
    },
    QuestionType.UNKNOWN: {
        "retrieval_strategy": RetrievalStrategy.FULL_PIPELINE,
        "requires_rewrite": True,
        "requires_self_query": True,
        "requires_multi_query": True,
        "requires_parent_retrieval": True,
        "requires_context_compression": True,
    },
}


class Planner:
    """Analyse a question and produce an ``ExecutionPlan``.

    The plan is deterministic (rule-based) today.  To swap in an
    LLM-based planner, replace the body of ``create_plan`` — no other
    code in the agent needs to change.
    """

    def create_plan(self, question: str) -> ExecutionPlan:
        """Return the best execution strategy for *question*.

        On failure returns ``ExecutionPlan.default_plan()`` so the
        graph never stops.
        """
        t_start = time.monotonic()

        try:
            qtype = classify(question)
            config = _STRATEGY_MAP.get(qtype, _STRATEGY_MAP[QuestionType.UNKNOWN])

            plan = ExecutionPlan(
                question_type=qtype,
                retrieval_strategy=RetrievalStrategy(str(config["retrieval_strategy"])),
                requires_rewrite=bool(config["requires_rewrite"]),
                requires_self_query=bool(config["requires_self_query"]),
                requires_multi_query=bool(config["requires_multi_query"]),
                requires_parent_retrieval=bool(config["requires_parent_retrieval"]),
                requires_context_compression=bool(config["requires_context_compression"]),
                reasoning=_build_reasoning(qtype, question),
            )

            elapsed = (time.monotonic() - t_start) * 1000
            logger.info(
                "Planner — type=%s  strategy=%s  rewrite=%s  multi=%s  latency=%.0fms",
                qtype.value,
                plan.retrieval_strategy.value,
                plan.requires_rewrite,
                plan.requires_multi_query,
                elapsed,
            )

            return plan

        except Exception:
            logger.exception("Planner failed — using default plan")
            return ExecutionPlan.default_plan(question)


def _build_reasoning(qtype: QuestionType, question: str) -> str:
    """Generate a human-readable explanation of the classification."""
    return (
        f"Classified as '{qtype.value}' based on keyword/pattern analysis. "
        f"Question length: {len(question)} chars."
    )
