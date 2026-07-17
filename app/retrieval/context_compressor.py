"""Context compression layer.

Reduces the volume of text sent to the LLM while preserving the
information required to answer the user's question.  Operates on
already-reranked ``SearchResult`` lists.

Supports pluggable strategies via the ``BaseContextCompressor``
abstract interface.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from app.config.settings import settings
from app.embeddings.service import EmbeddingService

if TYPE_CHECKING:
    from app.ingestion.models import SearchResult

logger = logging.getLogger(__name__)

# Rough heuristic: 4 chars ≈ 1 token for English text.
_CHARS_PER_TOKEN = 4


# ---------------------------------------------------------------------------
# abstract base
# ---------------------------------------------------------------------------


class BaseContextCompressor(ABC):
    """Interface for context compression strategies."""

    @abstractmethod
    def compress(
        self,
        question: str,
        results: list[SearchResult],
    ) -> list[SearchResult]:
        """Return a (possibly shorter) list of results."""
        ...


# ---------------------------------------------------------------------------
# similarity-based compressor
# ---------------------------------------------------------------------------


class SimilarityContextCompressor(BaseContextCompressor):
    """Remove redundant chunks and enforce a token budget.

    Strategy 1 — Redundancy removal:
        Computes cosine similarity between chunk embeddings.  Chunks
        whose similarity to an already-kept chunk exceeds
        *redundancy_threshold* are dropped.  Iteration order follows
        the input ranking (highest score first).

    Strategy 2 — Token budget:
        After deduplication, truncates the list to fit within
        *max_tokens* by keeping the highest-scoring chunks.

    Parameters
    ----------
    embedding_service:
        Used to embed chunk texts for similarity computation.
    redundancy_threshold:
        Cosine-similarity threshold above which a chunk is considered
        redundant (default from ``settings.redundancy_threshold``).
    max_tokens:
        Upper bound on total tokens across all kept chunks (default
        from ``settings.max_context_tokens``).
    """

    def __init__(
        self,
        embedding_service: EmbeddingService,
        *,
        redundancy_threshold: float | None = None,
        max_tokens: int | None = None,
    ) -> None:
        self._embed = embedding_service
        self._threshold = (
            redundancy_threshold
            if redundancy_threshold is not None
            else settings.redundancy_threshold
        )
        self._max_tokens = max_tokens or settings.max_context_tokens

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def compress(
        self,
        question: str,
        results: list[SearchResult],
    ) -> list[SearchResult]:
        """Apply redundancy removal, then token-budget truncation."""
        if len(results) <= 1:
            return results

        # -- 1. redundancy removal --------------------------------------
        deduped = self._remove_redundant(results)

        # -- 2. token budget enforcement --------------------------------
        trimmed = self._apply_token_budget(deduped)

        removed = len(results) - len(trimmed)
        logger.info(
            "Context compression — %d -> %d chunks  (%d removed  threshold=%.2f  max_tokens=%d)",
            len(results),
            len(trimmed),
            removed,
            self._threshold,
            self._max_tokens,
        )

        return trimmed

    # ------------------------------------------------------------------
    # internal
    # ------------------------------------------------------------------

    def _remove_redundant(
        self,
        results: list[SearchResult],
    ) -> list[SearchResult]:
        """Greedy deduplication: keep highest-score, discard near-duplicates."""
        if len(results) <= 1:
            return results

        texts = [r.chunk.text for r in results]
        embeddings = self._embed.embed_queries(texts)

        kept: list[SearchResult] = []
        kept_embeddings: list[list[float]] = []

        for i, result in enumerate(results):
            emb = embeddings[i]
            is_dup = False
            for kept_emb in kept_embeddings:
                if _cosine_similarity(emb, kept_emb) >= self._threshold:
                    is_dup = True
                    break
            if not is_dup:
                kept.append(result)
                kept_embeddings.append(emb)

        return kept

    def _apply_token_budget(
        self,
        results: list[SearchResult],
    ) -> list[SearchResult]:
        """Keep highest-score chunks until *max_tokens* is exhausted."""
        if not results:
            return []

        kept: list[SearchResult] = []
        total = 0
        for r in results:
            est = max(1, len(r.chunk.text) // _CHARS_PER_TOKEN)
            if total + est > self._max_tokens and kept:
                break
            kept.append(r)
            total += est

        return kept


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two already-normalised vectors."""
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    # Clamp to [-1, 1] to avoid floating-point drift
    return max(-1.0, min(1.0, dot))
