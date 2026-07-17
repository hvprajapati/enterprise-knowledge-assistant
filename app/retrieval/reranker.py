"""Cross-encoder reranking layer.

Re-scores candidate passages using a cross-encoder model so that the
most relevant passages surface to the top before prompt construction.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
from sentence_transformers import CrossEncoder

from app.config.settings import settings

if TYPE_CHECKING:
    from app.ingestion.models import SearchResult

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """Re-scores retrieved ``SearchResult`` objects with a cross-encoder.

    The model is loaded once at instantiation time.  Scoring is done in
    batches internally by the ``CrossEncoder.predict`` method.

    Parameters
    ----------
    model_name:
        Hugging Face model identifier.  Defaults to the value of
        ``Settings.reranker_model`` (``BAAI/bge-reranker-base``).
    batch_size:
        Number of (query, document) pairs scored per internal batch
        (default 32).
    """

    def __init__(
        self,
        model_name: str | None = None,
        batch_size: int = 32,
    ) -> None:
        self._model_name = model_name or settings.reranker_model
        self._batch_size = batch_size

        logger.info("Loading cross-encoder model: %s", self._model_name)
        try:
            self._model = CrossEncoder(self._model_name)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load cross-encoder model '{self._model_name}'"
            ) from exc
        logger.info("Cross-encoder model loaded successfully.")

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def rerank(
        self,
        query: str,
        results: list[SearchResult],
        top_k: int = 5,
    ) -> list[SearchResult]:
        """Re-score *results* and return the top-*k*.

        Parameters
        ----------
        query:
            Raw user question.  Must be non-empty after stripping.
        results:
            Candidate passages from the retriever.  An empty list
            returns an empty list immediately.
        top_k:
            Number of highest-scoring passages to keep (default 5).

        Returns
        -------
        list[SearchResult]
            New ``SearchResult`` objects with updated ``.score`` values,
            sorted descending, trimmed to *top_k*.

        Raises
        ------
        ValueError
            If *query* is empty or whitespace-only.
        RuntimeError
            If the underlying model fails during prediction.
        """
        query = query.strip()
        if not query:
            raise ValueError("query must be a non-empty string")

        if not results:
            logger.debug("rerank called with empty results — returning []")
            return []

        logger.info(
            "Reranking %d results for query: %s", len(results), query[:80]
        )

        # -- build (query, chunk_text) pairs ---------------------------
        pairs = [(query, result.chunk.text) for result in results]

        # -- predict relevance scores ----------------------------------
        try:
            scores: np.ndarray = self._model.predict(
                pairs,
                batch_size=self._batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Cross-encoder prediction failed for query: {query[:80]}"
            ) from exc

        # -- create new results with updated scores (no mutation) ------
        reranked = [
            result.model_copy(update={"score": float(score)})
            for result, score in zip(results, scores, strict=True)
        ]

        reranked.sort(key=lambda r: r.score, reverse=True)

        top = reranked[:top_k]

        logger.info(
            "Reranking complete — %d results → top %d (best score=%.4f)",
            len(results),
            len(top),
            top[0].score if top else 0.0,
        )

        return top
