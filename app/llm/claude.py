"""Claude (Anthropic) provider implementation."""

from __future__ import annotations

import logging
import time

import anthropic

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


class ClaudeLLM(BaseLLM):
    """LLM provider backed by Anthropic Claude models.

    Parameters
    ----------
    settings:
        Application settings — ``claude_api_key``, ``llm_model_name``,
        ``llm_timeout``, ``llm_max_retries`` are read from here.
    """

    def __init__(self, settings: Settings) -> None:
        self._model = settings.llm_model_name
        self._timeout = float(settings.llm_timeout)
        self._max_retries = settings.llm_max_retries

        if not settings.claude_api_key:
            raise LLMAuthenticationError(
                "Claude API key is not configured. Set CLAUDE_API_KEY in .env."
            )

        self._client = anthropic.Anthropic(
            api_key=settings.claude_api_key,
            timeout=self._timeout,
            max_retries=self._max_retries,
        )

        logger.info(
            "ClaudeLLM initialised — model=%s timeout=%.0fs max_retries=%d",
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

        payload: dict[str, object] = {
            "model": self._model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system_prompt:
            payload["system"] = system_prompt

        logger.info("Claude request started — model=%s max_tokens=%d", self._model, max_tokens)

        try:
            response = self._client.messages.create(**payload)
        except anthropic.AuthenticationError as exc:
            raise LLMAuthenticationError(str(exc)) from exc
        except anthropic.RateLimitError as exc:
            raise LLMRateLimitError(str(exc)) from exc
        except anthropic.APITimeoutError as exc:
            raise LLMTimeoutError(str(exc)) from exc
        except anthropic.APIConnectionError as exc:
            raise LLMNetworkError(str(exc)) from exc
        except anthropic.APIStatusError as exc:
            raise LLMProviderError(
                f"Claude returned HTTP {exc.status_code}: {exc.message}"
            ) from exc
        except Exception as exc:
            raise LLMProviderError(str(exc)) from exc

        elapsed = time.monotonic() - started

        usage = response.usage
        input_tokens = usage.input_tokens if usage else "?"
        output_tokens = usage.output_tokens if usage else "?"

        logger.info(
            "Claude request completed — latency=%.2fs input_tokens=%s output_tokens=%s",
            elapsed,
            input_tokens,
            output_tokens,
        )

        return response.content[0].text  # type: ignore[no-any-return]
