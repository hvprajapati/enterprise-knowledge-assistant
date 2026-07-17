"""Prompt construction layer.

Assembles a grounded RAG prompt from a user question and retrieved
context passages.  This module **never** calls an LLM — it only produces
the string that will later be sent to one.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from app.config.settings import settings

if TYPE_CHECKING:
    from app.ingestion.models import SearchResult

logger = logging.getLogger(__name__)

# Separator placed between context passages.
_PASSAGE_SEPARATOR = "\n\n" + "-" * 60 + "\n\n"


class PromptBuilder:
    """Builds a production-quality RAG prompt.

    Configurable values are read from ``Settings`` but can be
    overridden per instance via constructor parameters.

    Parameters
    ----------
    max_context_chunks:
        Safety cap on the number of passages included in the prompt.
        Defaults to ``Settings.prompt_max_context_chunks`` (20).
    system_text:
        System-level instructions.  Defaults to
        ``Settings.prompt_system_text``.
    no_context_text:
        Text used when *context* is empty.  Defaults to
        ``Settings.prompt_no_context_text``.
    """

    def __init__(
        self,
        max_context_chunks: int | None = None,
        system_text: str | None = None,
        no_context_text: str | None = None,
    ) -> None:
        self._max_chunks = max_context_chunks or settings.prompt_max_context_chunks
        self._system = system_text or settings.prompt_system_text
        self._no_context = no_context_text or settings.prompt_no_context_text

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def build_prompt(
        self,
        question: str,
        context: list[SearchResult],
    ) -> str:
        """Assemble the full RAG prompt.

        Parameters
        ----------
        question:
            The raw user question.
        context:
            Ranked retrieval results (may be empty).  Only the first
            ``max_context_chunks`` are used.

        Returns
        -------
        str
            A complete prompt string ready for an LLM.
        """
        logger.info(
            "Building prompt — question=%d chars, context chunks=%d",
            len(question),
            len(context),
        )

        if not context:
            return self._build_no_context_prompt(question)

        # Safety cap
        passages = context[: self._max_chunks]

        # -- format each retrieved passage ------------------------------
        formatted = self._format_passages(passages)

        # -- assemble final prompt -------------------------------------
        prompt = self._assemble(question, formatted)

        logger.info(
            "Prompt built — %d passages, %d chars total",
            len(passages),
            len(prompt),
        )

        return prompt

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------

    def _format_passages(
        self,
        results: list[SearchResult],
    ) -> str:
        """Render each ``SearchResult`` as a labelled passage block."""
        blocks: list[str] = []

        for i, result in enumerate(results, start=1):
            chunk = result.chunk
            page = str(chunk.page_number) if chunk.page_number is not None else "N/A"

            blocks.append(
                f"[{i}]  Source : {chunk.metadata.filename}\n"
                f"     Page   : {page}\n"
                f"     Score  : {result.score:.4f}\n"
                f"\n"
                f"{chunk.text}"
            )

        return _PASSAGE_SEPARATOR.join(blocks)

    def _assemble(self, question: str, context_block: str) -> str:
        """Combine system prompt, context, question, and answer preamble."""
        return (
            f"{self._system}\n"
            f"\n"
            f"{'=' * 60}\n"
            f"\n"
            f"{context_block}\n"
            f"\n"
            f"{'=' * 60}\n"
            f"Question:\n"
            f"{question}\n"
            f"\n"
            f"{'=' * 60}\n"
            f"Answer:\n"
        )

    def _build_no_context_prompt(self, question: str) -> str:
        """Return a prompt that tells the LLM to report no results."""
        logger.warning("No context passages available — building empty-context prompt.")

        return (
            f"{self._no_context}\n"
            f"\n"
            f"{'=' * 60}\n"
            f"User question (for reference):\n"
            f"{question}\n"
            f"\n"
            f"{'=' * 60}\n"
            f"Answer:\n"
        )
