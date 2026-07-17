from __future__ import annotations

import logging
from pathlib import Path

import faiss
import numpy as np

from app.ingestion.models import DocumentChunk, SearchResult
from app.vectorstore.metadata_store import MetadataStore

logger = logging.getLogger(__name__)


class FAISSVectorStore:
    """Stores embeddings inside a FAISS index."""

    def __init__(
        self,
        dimension: int,
    ) -> None:
        self.index: faiss.Index = faiss.IndexFlatIP(dimension)
        self.metadata_store = MetadataStore()

    def load_index(self, path: Path) -> None:
        """Replace the in-memory index with one loaded from disk."""
        self.index = faiss.read_index(str(path))

    def add(
        self,
        embedding: list[float],
        chunk: DocumentChunk,
    ) -> int:

        vector = np.asarray(
            [embedding],
            dtype="float32",
        )

        vector_id = self.index.ntotal

        self.index.add(vector)

        self.metadata_store.add(
            vector_id,
            chunk,
        )

        return vector_id

    def search(
        self,
        embedding: list[float],
        top_k: int = 5,
    ) -> list[SearchResult]:

        vector = np.asarray(
            [embedding],
            dtype="float32",
        )

        scores, indices = self.index.search(
            vector,
            top_k,
        )

        results: list[SearchResult] = []

        for score, idx in zip(scores[0], indices[0], strict=True):
            if idx == -1:
                continue

            try:
                chunk = self.metadata_store.get(idx)
            except KeyError:
                logger.warning(
                    "FAISS returned vector_id=%d but no chunk found in metadata store — skipping.",
                    idx,
                )
                continue

            results.append(
                SearchResult(
                    chunk=chunk,
                    score=float(score),
                )
            )

        return results
