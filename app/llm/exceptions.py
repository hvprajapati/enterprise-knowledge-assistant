"""Application-level exceptions for the LLM abstraction layer.

Every provider raises these exception types so callers never need to
catch provider-specific errors.
"""

from __future__ import annotations


class LLMError(Exception):
    """Base exception for all LLM-related failures."""


class LLMAuthenticationError(LLMError):
    """Invalid, expired, or missing API key."""


class LLMRateLimitError(LLMError):
    """The provider rate-limited the request."""


class LLMTimeoutError(LLMError):
    """The request exceeded the configured timeout."""


class LLMNetworkError(LLMError):
    """A network-level failure (DNS, connection refused, etc.)."""


class LLMProviderError(LLMError):
    """The provider returned an error response (4xx / 5xx)."""
