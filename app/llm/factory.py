"""Provider factory — creates the right LLM from configuration.

The rest of the application imports **only** this module.  It never
knows which concrete provider is active.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from app.llm.base import BaseLLM
from app.llm.exceptions import LLMProviderError

if TYPE_CHECKING:
    from app.config.settings import Settings

logger = logging.getLogger(__name__)

# Registry of supported providers.  Add a new entry here when you add a
# new implementation — no other module needs to change.
_PROVIDER_REGISTRY: dict[str, str] = {
    "claude": "app.llm.claude.ClaudeLLM",
    "openai": "app.llm.openai.OpenAILLM",
    "gemini": "app.llm.gemini.GeminiLLM",
}


class LLMFactory:
    """Create a ``BaseLLM`` instance based on ``Settings.llm_provider``.

    Usage::

        from app.config.settings import settings
        from app.llm.factory import LLMFactory

        llm = LLMFactory(settings).create()
        answer = llm.generate(prompt)

    Swap providers by changing ``LLM_PROVIDER`` in ``.env`` — zero
    code changes required.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def create(self) -> BaseLLM:
        """Return a ready-to-use ``BaseLLM`` for the configured provider.

        Raises
        ------
        LLMProviderError
            If ``settings.llm_provider`` is not in the registry or the
            class cannot be imported.
        """
        provider = self._settings.llm_provider.lower()

        qualname = _PROVIDER_REGISTRY.get(provider)
        if qualname is None:
            valid = ", ".join(sorted(_PROVIDER_REGISTRY))
            raise LLMProviderError(
                f"Unknown LLM provider '{provider}'. "
                f"Valid choices: {valid}."
            )

        module_name, class_name = qualname.rsplit(".", 1)

        try:
            import importlib

            module = importlib.import_module(module_name)
            cls = getattr(module, class_name)
        except Exception as exc:
            raise LLMProviderError(
                f"Failed to load provider '{provider}' ({qualname}): {exc}"
            ) from exc

        logger.info("Creating LLM provider: %s (%s)", provider, qualname)
        return cls(self._settings)  # type: ignore[no-any-return]
