"""Abstract base class for LLM providers.

Every provider must implement ``generate``.  This interface is the
**only** contract the rest of the application depends on.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class BaseLLM(ABC):
    """Vendor-independent LLM interface.

    Subclasses implement the ``generate`` method for a specific
    provider (Claude, OpenAI, Gemini, …).
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
        """Run inference and return the completion text.

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

        Returns
        -------
        str
            The model's text response.

        Raises
        ------
        LLMError
            One of the subclasses in :mod:`app.llm.exceptions`.
        """
        ...
