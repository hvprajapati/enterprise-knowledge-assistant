"""Retrieval Orchestrator — builds a ``RetrievalPlan`` from a question."""

from __future__ import annotations

import logging
import time

from app.retrieval.orchestrator.classifier import RetrievalClassifier
from app.retrieval.orchestrator.models import QuestionType, RetrievalPlan

logger = logging.getLogger(__name__)

# Mapping: QuestionType → which stages to enable
_PLAN_TEMPLATES: dict[QuestionType, dict[str, bool | int]] = {
    QuestionType.SIMPLE: {
        "rewrite_query": False,
        "use_self_query": False,
        "use_multi_query": False,
        "use_hybrid_search": False,
        "use_parent_document": False,
        "use_context_compression": False,
        "vector_top_k": 30,
        "bm25_top_k": 0,
    },
    QuestionType.KEYWORD: {
        "rewrite_query": False,
        "use_self_query": False,
        "use_multi_query": False,
        "use_hybrid_search": True,       # BM25 helps keyword queries
        "use_parent_document": False,
        "use_context_compression": False,
        "vector_top_k": 40,
        "bm25_top_k": 40,
    },
    QuestionType.METADATA: {
        "rewrite_query": False,
        "use_self_query": True,
        "use_multi_query": False,
        "use_hybrid_search": True,
        "use_parent_document": False,
        "use_context_compression": False,
        "vector_top_k": 50,
        "bm25_top_k": 50,
    },
    QuestionType.AMBIGUOUS: {
        "rewrite_query": True,
        "use_self_query": False,
        "use_multi_query": False,
        "use_hybrid_search": False,
        "use_parent_document": False,
        "use_context_compression": False,
        "vector_top_k": 30,
        "bm25_top_k": 0,
    },
    QuestionType.BROAD: {
        "rewrite_query": True,
        "use_self_query": False,
        "use_multi_query": True,
        "use_hybrid_search": True,
        "use_parent_document": False,
        "use_context_compression": True,
        "vector_top_k": 50,
        "bm25_top_k": 50,
    },
    QuestionType.COMPARISON: {
        "rewrite_query": True,
        "use_self_query": False,
        "use_multi_query": True,
        "use_hybrid_search": True,
        "use_parent_document": True,
        "use_context_compression": True,
        "vector_top_k": 50,
        "bm25_top_k": 50,
    },
    QuestionType.TROUBLESHOOTING: {
        "rewrite_query": True,
        "use_self_query": False,
        "use_multi_query": True,
        "use_hybrid_search": True,
        "use_parent_document": True,
        "use_context_compression": True,
        "vector_top_k": 50,
        "bm25_top_k": 50,
    },
    QuestionType.FACTUAL: {
        "rewrite_query": False,
        "use_self_query": False,
        "use_multi_query": False,
        "use_hybrid_search": False,
        "use_parent_document": False,
        "use_context_compression": False,
        "vector_top_k": 50,
        "bm25_top_k": 0,
    },
}


# ---------------------------------------------------------------------------
# orchestrator
# ---------------------------------------------------------------------------


class RetrievalOrchestrator:
    """Build a ``RetrievalPlan`` tailored to the user's question.

    Parameters
    ----------
    classifier:
        Question classifier.  When ``None`` a default
        ``RetrievalClassifier`` is created.
    """

    def __init__(
        self,
        classifier: RetrievalClassifier | None = None,
    ) -> None:
        self._classifier = classifier or RetrievalClassifier()

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def plan(self, question: str) -> RetrievalPlan:
        """Return a plan describing which stages to execute.

        On failure returns ``RetrievalPlan.full()`` so the pipeline
        always produces an answer.
        """
        t_start = time.monotonic()

        try:
            qtype = self._classifier.classify(question)
            template = _PLAN_TEMPLATES.get(qtype, _PLAN_TEMPLATES[QuestionType.FACTUAL])

            plan = RetrievalPlan(
                question_type=qtype,
                rerank=True,
                rewrite_query=bool(template.get("rewrite_query", False)),
                use_self_query=bool(template.get("use_self_query", False)),
                use_multi_query=bool(template.get("use_multi_query", False)),
                use_hybrid_search=bool(template.get("use_hybrid_search", False)),
                use_parent_document=bool(template.get("use_parent_document", False)),
                use_context_compression=bool(template.get("use_context_compression", False)),
                vector_top_k=int(template.get("vector_top_k", 50)),
                bm25_top_k=int(template.get("bm25_top_k", 0)),
            )

            elapsed = (time.monotonic() - t_start) * 1000
            logger.info(
                "Retrieval plan — type=%s  stages=%s  latency=%.0fms",
                qtype.value,
                plan.stages_enabled,
                elapsed,
            )

            return plan

        except Exception:
            logger.exception("Orchestration failed — using full pipeline")
            return RetrievalPlan.full()
