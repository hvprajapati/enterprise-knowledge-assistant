"""Multi-query retrieval generation.

Produces 3–5 semantically diverse search queries from a single user
question.  Each query explores different wording, synonyms, and
technical perspectives to maximise retrieval recall.
"""

from __future__ import annotations

import logging
import time

from app.config.settings import settings
from app.llm.base import BaseLLM

logger = logging.getLogger(__name__)


class MultiQueryGenerator:
    """Generate multiple retrieval-oriented queries from one question.

    Parameters
    ----------
    llm:
        Vendor-independent LLM used for query generation.
    system_prompt:
        Override the default generation prompt.  When ``None``,
        ``settings.multi_query_system_prompt`` is used.
    max_variants:
        Maximum number of queries to generate (default from
        ``settings.multi_query_max_variants``).
    enabled:
        When ``False``, ``generate`` returns a single-element list
        containing only *question*.  Defaults to
        ``settings.multi_query_enabled``.
    """

    def __init__(
        self,
        llm: BaseLLM,
        *,
        system_prompt: str | None = None,
        max_variants: int | None = None,
        enabled: bool | None = None,
    ) -> None:
        self._llm = llm
        self._system_prompt = system_prompt or settings.multi_query_system_prompt
        self._max_variants = max_variants or settings.multi_query_max_variants
        self._enabled = enabled if enabled is not None else settings.multi_query_enabled

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def generate(self, question: str, max_queries: int | None = None) -> list[str]:
        """Return a list of semantically diverse search queries.

        On failure returns ``[question]`` so the pipeline continues
        with at least the original question.
        """
        if not self._enabled:
            return [question]

        max_q = max_queries or self._max_variants

        t_start = time.monotonic()
        logger.info(
            "Multi-query generation started — input=%d chars  max_queries=%d",
            len(question),
            max_q,
        )

        try:
            # Ask the LLM to produce max_q + 1 lines in case one is blank
            prompt = (
                f"Generate {max_q} different search queries for the "
                f"following question:\n\n{question}"
            )
            raw = self._llm.generate(
                prompt=prompt,
                system_prompt=self._system_prompt,
                temperature=0.3,
                max_tokens=256,
            )

            # Parse lines: strip, drop blanks, keep unique, limit
            queries: list[str] = [question]  # always include the original
            seen: set[str] = {question.lower()}
            for line in raw.strip().splitlines():
                cleaned = line.strip().lstrip("-•*0123456789. ").strip()
                if cleaned and cleaned.lower() not in seen:
                    queries.append(cleaned)
                    seen.add(cleaned.lower())
                if len(queries) >= max_q:
                    break

            elapsed = (time.monotonic() - t_start) * 1000
            logger.info(
                "Multi-query generation completed — %d queries  latency=%.0fms",
                len(queries),
                elapsed,
            )

            return queries if queries else [question]

        except Exception:
            logger.exception(
                "Multi-query generation failed — falling back to single query"
            )
            return [question]
