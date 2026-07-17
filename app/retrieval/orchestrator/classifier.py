"""Question classifier — currently rule-based, designed for future LLM replacement."""

from __future__ import annotations

import logging

from app.retrieval.orchestrator.models import QuestionType
from app.retrieval.orchestrator.rules import classify as _rule_classify

logger = logging.getLogger(__name__)


class RetrievalClassifier:
    """Classify a user question into a ``QuestionType``.

    Currently uses deterministic keyword/pattern rules.  The interface
    is deliberately minimal so an LLM-based classifier can be swapped
    in later without changing any caller code.

    Parameters
    ----------
    use_llm:
        When ``True`` and an *llm* is provided, classification is
        delegated to the LLM.  Currently not implemented — reserved
        for future enhancement.
    """

    def __init__(self, use_llm: bool = False) -> None:
        self._use_llm = use_llm

    def classify(self, question: str) -> QuestionType:
        """Return the question category.

        Falls back to ``QuestionType.FACTUAL`` on any error so
        callers never receive an exception.
        """
        try:
            qtype = _rule_classify(question)
            logger.debug("Classified as %s: %s", qtype.value, question[:80])
            return qtype
        except Exception:
            logger.exception("Classification failed — defaulting to FACTUAL")
            return QuestionType.FACTUAL
