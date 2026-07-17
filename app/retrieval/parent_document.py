"""Parent Document Retrieval.

Expands each retrieved chunk with its neighbours from the same
document, improving context continuity while preserving retrieval
precision.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING
from uuid import UUID

from app.config.settings import settings

if TYPE_CHECKING:
    from app.ingestion.models import DocumentChunk, SearchResult
    from app.storage.repository import ChunkRepository

logger = logging.getLogger(__name__)


class ParentDocumentRetriever:
    """Expand retrieved chunks with neighbouring chunks from the same document.

    For each chunk in *results*, the retriever looks up
    ``chunk_index ± window_size`` chunks from SQLite via the
    ``ChunkRepository``.  Neighbouring chunks are merged, deduplicated
    by ``chunk_id``, and returned in the original ranking order.

    Parameters
    ----------
    repository:
        SQLite-backed chunk store used to look up neighbours.
    window_size:
        Number of chunks to include on each side (default from
        ``settings.parent_window_size``).
    max_chunks:
        Hard ceiling on total chunks after expansion (default from
        ``settings.max_parent_chunks``).
    """

    def __init__(
        self,
        repository: ChunkRepository,
        *,
        window_size: int | None = None,
        max_chunks: int | None = None,
    ) -> None:
        self._repo = repository
        self._window = window_size if window_size is not None else settings.parent_window_size
        self._max = max_chunks if max_chunks is not None else settings.max_parent_chunks

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def expand(
        self,
        results: list[SearchResult],
        window_size: int | None = None,
    ) -> list[SearchResult]:
        """Return *results* expanded with neighbouring chunks.

        Parameters
        ----------
        results:
            Ranked list of retrieved chunks (post-rerank).
        window_size:
            Override the configured window size for this call.

        Returns
        -------
        list[SearchResult]
            Expanded list, deduplicated, preserving original order,
            capped at ``max_chunks``.  On failure returns *results*
            unchanged.
        """
        if not results:
            return results

        window = window_size if window_size is not None else self._window

        t_start = time.monotonic()

        try:
            expanded = self._expand(results, window)
        except Exception:
            logger.exception(
                "Parent document expansion failed — returning original %d chunks",
                len(results),
            )
            return results

        elapsed = (time.monotonic() - t_start) * 1000
        logger.info(
            "Parent document expansion — %d -> %d chunks  (window=%d  latency=%.0fms)",
            len(results),
            len(expanded),
            window,
            elapsed,
        )

        return expanded

    # ------------------------------------------------------------------
    # internal
    # ------------------------------------------------------------------

    def _expand(
        self,
        results: list[SearchResult],
        window: int,
    ) -> list[SearchResult]:
        from app.ingestion.models import SearchResult

        # 1. Group by document_id, collect (chunk_index, original_score)
        doc_indices: dict[str, list[tuple[int, float]]] = {}
        for r in results:
            doc_id = str(r.chunk.document_id)
            doc_indices.setdefault(doc_id, []).append((r.chunk.chunk_index, r.score))

        # 2. Load all chunks for each document once (batched by doc)
        doc_chunks: dict[str, dict[int, DocumentChunk]] = {}
        for doc_id in doc_indices:
            try:
                doc_chunks[doc_id] = self._repo.get_chunks_by_document(
                    UUID(doc_id)
                )
            except Exception:
                logger.warning("Failed to load chunks for document %s", doc_id)

        # 3. Build the expanded set
        seen: set[UUID] = set()
        ordered: list[SearchResult] = []

        for r in results:
            doc_id = str(r.chunk.document_id)
            chunks = doc_chunks.get(doc_id, {})
            if not chunks:
                # No neighbour data — keep the original chunk
                if r.chunk.chunk_id not in seen:
                    seen.add(r.chunk.chunk_id)
                    ordered.append(r)
                continue

            # Collect the window: [idx-window, idx+window]
            for offset in range(-window, window + 1):
                neighbour_idx = r.chunk.chunk_index + offset
                neighbour = chunks.get(neighbour_idx)
                if neighbour is None:
                    continue
                cid = neighbour.chunk_id
                if cid not in seen:
                    seen.add(cid)
                    # Use original score for the anchor chunk, keep
                    # nearby chunks ordered but assign score 0
                    sr = SearchResult(
                        chunk=neighbour,
                        score=r.score if offset == 0 else 0.0,
                    )
                    ordered.append(sr)

        # 4. Cap
        return ordered[: self._max]
