"""OpenAI provider implementation."""

from __future__ import annotations

import logging
import time

import openai
from openai import OpenAI

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


class OpenAILLM(BaseLLM):
    """LLM provider backed by OpenAI models (GPT-4o, etc.).

    Parameters
    ----------
    settings:
        Application settings — ``openai_api_key``, ``llm_model_name``,
        ``llm_timeout``, ``llm_max_retries`` are read from here.
    """

    def __init__(self, settings: Settings) -> None:
        self._model = settings.llm_model_name
        self._timeout = float(settings.llm_timeout)
        self._max_retries = settings.llm_max_retries

        if not settings.openai_api_key:
            raise LLMAuthenticationError(
                "OpenAI API key is not configured. Set OPENAI_API_KEY in .env."
            )

        self._client = OpenAI(
            api_key=settings.openai_api_key,
            timeout=self._timeout,
            max_retries=self._max_retries,
        )

        logger.info(
            "OpenAILLM initialised — model=%s timeout=%.0fs max_retries=%d",
            self._model,
            self._timeout,
            self._max_retries,
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

        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        logger.info(
            "OpenAI request started — model=%s max_tokens=%d", self._model, max_tokens
        )

        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except openai.AuthenticationError as exc:
            raise LLMAuthenticationError(str(exc)) from exc
        except openai.RateLimitError as exc:
            raise LLMRateLimitError(str(exc)) from exc
        except openai.APITimeoutError as exc:
            raise LLMTimeoutError(str(exc)) from exc
        except openai.APIConnectionError as exc:
            raise LLMNetworkError(str(exc)) from exc
        except openai.APIStatusError as exc:
            raise LLMProviderError(
                f"OpenAI returned HTTP {exc.status_code}: {exc.message}"
            ) from exc
        except Exception as exc:
            raise LLMProviderError(str(exc)) from exc

        elapsed = time.monotonic() - started

        usage = response.usage
        prompt_tokens = usage.prompt_tokens if usage else "?"
        completion_tokens = usage.completion_tokens if usage else "?"

        logger.info(
            "OpenAI request completed — latency=%.2fs prompt_tokens=%s completion_tokens=%s",
            elapsed,
            prompt_tokens,
            completion_tokens,
        )

        return response.choices[0].message.content or ""
