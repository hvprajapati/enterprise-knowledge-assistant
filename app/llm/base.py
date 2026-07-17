"""Abstract base class for LLM providers.

Every provider must implement ``generate`` and ``stream_generate``.
This interface is the **only** contract the rest of the application
depends on.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator


class BaseLLM(ABC):
    """Vendor-independent LLM interface.

    Subclasses implement the ``generate`` and ``stream_generate``
    methods for a specific provider (Claude, OpenAI, Gemini, …).
    """

    @abstractmethod
    def generate(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> str:
        """Run inference and return the complete completion text."""
        ...

    @abstractmethod
    def stream_generate(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> Iterator[str]:
        """Run inference and yield tokens as they are produced.

        Each yielded value is a small chunk of text (a few tokens).
        Concatenating all chunks should produce the same text as
        ``generate`` would return for the same inputs.

        Parameters
        ----------
        prompt:
            The user message / RAG prompt to send.
        system_prompt:
            Optional system-level instructions.
        temperature:
            Sampling temperature (0.0 = deterministic).
        max_tokens:
            Upper bound on output tokens.

        Yields
        ------
        str
            Incremental text chunks from the model.

        Raises
        ------
        LLMError
            One of the subclasses in :mod:`app.llm.exceptions`.
        """
        ...
