"""BM25 (sparse / keyword) retrieval.

Builds a BM25 index from document chunks so that keyword-based
retrieval can complement dense (FAISS) retrieval in a hybrid pipeline.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from rank_bm25 import BM25Okapi

if TYPE_CHECKING:
    from app.ingestion.models import DocumentChunk, SearchResult

logger = logging.getLogger(__name__)


class BM25Retriever:
    """Sparse retriever powered by BM25 (Okapi BM25).

    The index is built once at construction time from the full set of
    document chunks.  For production use with very large corpora
    consider persisting the tokenized corpus.

    Parameters
    ----------
    chunks:
        All document chunks that should be searchable via BM25.
    """

    def __init__(self, chunks: list[DocumentChunk]) -> None:
        self._chunks = chunks
        self._tokenized = [_tokenize(c.text) for c in chunks]
        self._index = BM25Okapi(self._tokenized)

        logger.info(
            "BM25 index built — %d documents  avg_tokens=%.0f",
            len(chunks),
            sum(len(t) for t in self._tokenized) / max(len(chunks), 1),
        )

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        top_k: int = 50,
    ) -> list[SearchResult]:
        """Return the top-*k* keyword-matching chunks.

        Parameters
        ----------
        query:
            Raw query text (will be tokenized).
        top_k:
            Maximum number of results to return.

        Returns
        -------
        list[SearchResult]
            Results with BM25 scores (higher = more relevant).
        """
        from app.ingestion.models import SearchResult

        tokenized_query = _tokenize(query)
        scores = self._index.get_scores(tokenized_query)

        # Build (index, score) pairs, sort descending, take top_k
        indexed = sorted(
            enumerate(scores), key=lambda x: x[1], reverse=True
        )
        top = indexed[:top_k]

        return [
            SearchResult(chunk=self._chunks[idx], score=float(score))
            for idx, score in top
            if score > 0
        ]


# ------------------------------------------------------------------
# internal helpers
# ------------------------------------------------------------------


def _tokenize(text: str) -> list[str]:
    """Simple whitespace + lowercase tokenizer."""
    return text.lower().split()
