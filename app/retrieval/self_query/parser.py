"""LLM-powered self-query parser.

Uses the configured LLM with a structured prompt to extract metadata
filters from natural-language questions.
"""

from __future__ import annotations

import logging
import time

from app.llm.base import BaseLLM
from app.retrieval.self_query.models import StructuredQuery
from app.retrieval.self_query.validator import FilterValidator

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# system prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a metadata-extraction assistant for a RAG system.

Given a user question, extract any explicit metadata constraints and
rewrite the question WITHOUT those constraints.

Return ONLY a valid JSON object with exactly these keys:

{
  "rewritten_query": "<the content-only question>",
  "metadata_filters": {
    "filename": null,
    "source": null,
    "document_type": null,
    "tags": null,
    "extension": null,
    "year": null,
    "author": null
  }
}

Rules:
1. Set a filter value ONLY when the user EXPLICITLY mentions it.
2. Leave unmentioned filters as null.
3. "tags" accepts a single string (comma-separated for multiple).
4. "year" should be extracted as a string (e.g. "2025").
5. rewritten_query must contain only the content part of the question.
6. Output ONLY the JSON. No markdown, no preamble, no explanation.

Examples:

Input: "Show me AWS documents from 2025"
Output: {"rewritten_query":"Show me documents from 2025","metadata_filters":{"filename":null,"source":null,"document_type":"aws","tags":null,"extension":null,"year":"2025","author":null}}

Input: "What is FAISS?"
Output: {"rewritten_query":"What is FAISS?","metadata_filters":{"filename":null,"source":null,"document_type":null,"tags":null,"extension":null,"year":null,"author":null}}

Input: "Find papers by Smith about neural networks pdf"
Output: {"rewritten_query":"Find papers about neural networks","metadata_filters":{"filename":null,"source":null,"document_type":null,"tags":null,"extension":".pdf","year":null,"author":"Smith"}}
"""  # noqa: E501


# ---------------------------------------------------------------------------
# parser
# ---------------------------------------------------------------------------


class SelfQueryParser:
    """Extract structured metadata filters from natural-language questions.

    Parameters
    ----------
    llm:
        Vendor-independent LLM used for extraction.
    validator:
        Validates and sanitises the LLM output.  Created with defaults
        when ``None``.
    system_prompt:
        Override the default extraction prompt.
    """

    def __init__(
        self,
        llm: BaseLLM,
        *,
        validator: FilterValidator | None = None,
        system_prompt: str | None = None,
    ) -> None:
        self._llm = llm
        self._validator = validator or FilterValidator()
        self._system_prompt = system_prompt or _SYSTEM_PROMPT

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def parse(self, question: str) -> StructuredQuery:
        """Extract metadata filters from *question*.

        On **any** failure (LLM error, invalid JSON, validation error)
        the original question is returned with empty filters so the
        pipeline continues uninterrupted.
        """
        t_start = time.monotonic()
        logger.info("Self-query parsing started — input=%d chars", len(question))

        try:
            raw = self._llm.generate(
                prompt=question,
                system_prompt=self._system_prompt,
                temperature=0.0,
                max_tokens=256,
            )

            parsed = self._validator.validate(raw, question)

            elapsed = (time.monotonic() - t_start) * 1000
            active = {k: v for k, v in parsed.metadata_filters.items() if v is not None}
            logger.info(
                "Self-query parsed — rewritten=%d chars  active_filters=%s  latency=%.0fms",
                len(parsed.rewritten_query),
                active or "(none)",
                elapsed,
            )

            return parsed

        except Exception:
            logger.exception(
                "Self-query parsing failed — returning original question with no filters"
            )
            return StructuredQuery.empty(question)
