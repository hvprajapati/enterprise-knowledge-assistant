"""Hybrid retrieval — combines dense (FAISS) and sparse (BM25) results.

Uses Reciprocal Rank Fusion (RRF) to merge the two ranked lists into
a single, relevance-ordered result set.  Duplicates are merged by
summing their RRF scores.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from app.config.settings import settings
from app.retrieval.bm25 import BM25Retriever
from app.retrieval.retriever import Retriever

if TYPE_CHECKING:
    from app.ingestion.models import SearchResult

logger = logging.getLogger(__name__)


class HybridRetriever:
    """Combine dense and sparse retrieval with RRF fusion.

    Parameters
    ----------
    dense:
        FAISS-backed ``Retriever`` for semantic / dense search.
    bm25:
        ``BM25Retriever`` for keyword / sparse search.
    rrf_k:
        RRF constant (default from ``settings.rrf_k``).  Higher values
        reduce the influence of high-ranked individual results.
    dense_top_k:
        How many candidates to pull from FAISS per query.
    bm25_top_k:
        How many candidates to pull from BM25 per query.
    """

    def __init__(
        self,
        dense: Retriever,
        bm25: BM25Retriever,
        *,
        rrf_k: int | None = None,
        dense_top_k: int | None = None,
        bm25_top_k: int | None = None,
    ) -> None:
        self._dense = dense
        self._bm25 = bm25
        self._rrf_k = rrf_k if rrf_k is not None else settings.rrf_k
        self._dense_k = dense_top_k or settings.vector_top_k
        self._bm25_k = bm25_top_k or settings.bm25_top_k

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query_embedding: list[float],
        query_text: str,
        top_k: int = 50,
        *,
        metadata_filters: dict[str, str | None] | None = None,
    ) -> list[SearchResult]:
        """Run both retrievers, apply filters, and fuse via RRF.

        Parameters
        ----------
        query_embedding:
            Dense embedding vector for FAISS search.
        query_text:
            Raw text for BM25 keyword search.
        top_k:
            Number of fused results to return.
        metadata_filters:
            Optional dict of ``{field: value}``.  When provided,
            results from **both** retrievers are filtered before
            RRF fusion.
        """
        from app.retrieval.self_query.validator import apply_filters

        dense_ok = True
        sparse_ok = True

        # -- dense -------------------------------------------------------
        try:
            dense_results = self._dense.retrieve(query_embedding, top_k=self._dense_k)
            if metadata_filters:
                dense_results = apply_filters(
                    dense_results, metadata_filters
                )
            logger.debug(
                "Dense (FAISS) retrieved %d results (filtered)", len(dense_results)
            )
        except Exception:
            logger.exception("Dense retrieval failed")
            dense_results = []
            dense_ok = False

        # -- sparse ------------------------------------------------------
        try:
            sparse_results = self._bm25.retrieve(query_text, top_k=self._bm25_k)
            if metadata_filters:
                sparse_results = apply_filters(
                    sparse_results, metadata_filters
                )
            logger.debug(
                "Sparse (BM25) retrieved %d results (filtered)", len(sparse_results)
            )
        except Exception:
            logger.exception("Sparse retrieval failed")
            sparse_results = []
            sparse_ok = False

        # -- both failed → empty -----------------------------------------
        if not dense_ok and not sparse_ok:
            logger.error("Both dense and sparse retrieval failed")
            return []

        # -- fuse --------------------------------------------------------
        fused = _rrf_fuse(dense_results, sparse_results, k=self._rrf_k)

        logger.info(
            "Hybrid retrieval — dense=%d  sparse=%d  fused=%d  rrf_k=%d",
            len(dense_results),
            len(sparse_results),
            len(fused),
            self._rrf_k,
        )

        return fused[:top_k]


# ---------------------------------------------------------------------------
# RRF — Reciprocal Rank Fusion
# ---------------------------------------------------------------------------


def _rrf_fuse(
    dense: list[SearchResult],
    sparse: list[SearchResult],
    *,
    k: int = 60,
) -> list[SearchResult]:
    """Merge two ranked lists using Reciprocal Rank Fusion.

    For each unique chunk the RRF score is::

        Σ  1 / (k + rank)
       r∈{dense,sparse}

    where *rank* is 1-indexed.  The fused list is sorted descending by
    this aggregated score.

    Parameters
    ----------
    dense:
        Results from the dense retriever (already ordered by score).
    sparse:
        Results from the sparse retriever (already ordered by score).
    k:
        RRF constant — higher = less influence from top-ranked items.
    """
    from app.ingestion.models import SearchResult

    rrf_scores: dict[str, float] = {}
    chunk_map: dict[str, SearchResult] = {}

    for rank, result in enumerate(dense, start=1):
        cid = str(result.chunk.chunk_id)
        rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (k + rank)
        if cid not in chunk_map:
            chunk_map[cid] = result

    for rank, result in enumerate(sparse, start=1):
        cid = str(result.chunk.chunk_id)
        rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (k + rank)
        if cid not in chunk_map:
            chunk_map[cid] = result

    # Sort by RRF score descending
    sorted_ids = sorted(rrf_scores, key=rrf_scores.get, reverse=True)  # type: ignore[arg-type]

    duplicates = len(dense) + len(sparse) - len(sorted_ids)
    logger.debug(
        "RRF — %d raw  %d duplicates removed  %d fused",
        len(dense) + len(sparse),
        duplicates,
        len(sorted_ids),
    )

    return [
        SearchResult(chunk=chunk_map[cid].chunk, score=rrf_scores[cid])
        for cid in sorted_ids
    ]
