"""Query rewriting layer.

Improves raw user questions before they reach the embedding /
retrieval stage.  Uses the configured LLM with a dedicated system
prompt that instructs the model to expand, clarify, and de-ambiguate
without ever answering the question.

On any failure the original question is returned unchanged so that
the pipeline is never blocked by a rewriting error.
"""

from __future__ import annotations

import logging
import time

from app.config.settings import settings
from app.llm.base import BaseLLM

logger = logging.getLogger(__name__)


class QueryRewriter:
    """Expand and clarify user questions for better retrieval.

    Parameters
    ----------
    llm:
        Vendor-independent LLM used for rewriting.
    system_prompt:
        Override the default rewriting prompt.  When ``None``,
        ``settings.query_rewriter_system_prompt`` is used.
    enabled:
        When ``False``, ``rewrite`` returns the question unchanged.
        Defaults to ``settings.query_rewriter_enabled``.
    """

    def __init__(
        self,
        llm: BaseLLM,
        *,
        system_prompt: str | None = None,
        enabled: bool | None = None,
    ) -> None:
        self._llm = llm
        self._system_prompt = system_prompt or settings.query_rewriter_system_prompt
        self._enabled = enabled if enabled is not None else settings.query_rewriter_enabled

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def rewrite(self, question: str) -> str:
        """Return a rewritten version of *question*, or the original on failure.

        Behaviour is a no-op when ``enabled`` is ``False``.
        """
        if not self._enabled:
            return question

        if not question.strip():
            return question

        t_start = time.monotonic()
        logger.info("Query rewriting started — input=%d chars", len(question))

        try:
            result = self._llm.generate(
                prompt=question,
                system_prompt=self._system_prompt,
                temperature=0.0,
                max_tokens=256,
            )
            rewritten = result.strip()
            if not rewritten:
                logger.warning("Query rewriting returned empty — using original.")
                return question

            elapsed = (time.monotonic() - t_start) * 1000
            logger.info(
                "Query rewriting completed — latency=%.0fms  input=%d chars  output=%d chars",
                elapsed,
                len(question),
                len(rewritten),
            )
            return rewritten

        except Exception:
            logger.exception(
                "Query rewriting failed — returning original question (%d chars)",
                len(question),
            )
            return question
