"""Gemini (Google) provider implementation."""

from __future__ import annotations

import logging
import time

import google.generativeai as genai

from app.config.settings import Settings
from app.llm.base import BaseLLM
from app.llm.exceptions import (
    LLMAuthenticationError,
    LLMNetworkError,
    LLMProviderError,
    LLMRateLimitError,
    LLMTimeoutError,
)

logger = logging.getLogger(__name__)

# Map google-api-core exception types at module level so we can catch
# them even if the import fails (the package is optional).
try:
    from google.api_core import exceptions as google_exceptions
except ImportError:  # pragma: no cover
    google_exceptions = None


class GeminiLLM(BaseLLM):
    """LLM provider backed by Google Gemini models.

    Parameters
    ----------
    settings:
        Application settings — ``gemini_api_key``, ``llm_model_name``,
        ``llm_timeout`` are read from here.
    """

    def __init__(self, settings: Settings) -> None:
        self._model_name = settings.llm_model_name
        self._timeout = settings.llm_timeout

        if not settings.gemini_api_key:
            raise LLMAuthenticationError(
                "Gemini API key is not configured. Set GEMINI_API_KEY in .env."
            )

        genai.configure(api_key=settings.gemini_api_key)
        self._model = genai.GenerativeModel(self._model_name)

        logger.info(
            "GeminiLLM initialised — model=%s timeout=%ds",
            self._model_name,
            self._timeout,
        )

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def generate(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> str:
        started = time.monotonic()

        # Gemini combines system + user into the contents list
        contents: list[str] = []
        if system_prompt:
            contents.append(f"System: {system_prompt}\n\nUser: {prompt}")
        else:
            contents.append(prompt)

        generation_config = genai.types.GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
        )

        logger.info(
            "Gemini request started — model=%s max_tokens=%d",
            self._model_name,
            max_tokens,
        )

        try:
            response = self._model.generate_content(
                contents,
                generation_config=generation_config,
                request_options={"timeout": self._timeout},
            )
        except Exception as exc:
            if google_exceptions is not None:
                if isinstance(exc, google_exceptions.Unauthenticated):
                    raise LLMAuthenticationError(str(exc)) from exc
                if isinstance(exc, google_exceptions.ResourceExhausted):
                    raise LLMRateLimitError(str(exc)) from exc
                if isinstance(exc, google_exceptions.DeadlineExceeded):
                    raise LLMTimeoutError(str(exc)) from exc
                if isinstance(exc, google_exceptions.ServiceUnavailable):
                    raise LLMNetworkError(str(exc)) from exc
                if isinstance(exc, google_exceptions.GoogleAPIError):
                    raise LLMProviderError(str(exc)) from exc
            raise LLMProviderError(str(exc)) from exc

        elapsed = time.monotonic() - started

        usage = getattr(response, "usage_metadata", None)
        prompt_tokens = usage.prompt_token_count if usage else "?"
        candidates_tokens = usage.candidates_token_count if usage else "?"

        logger.info(
            "Gemini request completed — latency=%.2fs prompt_tokens=%s candidates_tokens=%s",
            elapsed,
            prompt_tokens,
            candidates_tokens,
        )

        return response.text  # type: ignore[no-any-return]
