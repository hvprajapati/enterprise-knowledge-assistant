import faiss
import numpy as np

from app.ingestion.models import DocumentChunk, SearchResult
from app.vectorstore.metadata_store import MetadataStore


class FAISSVectorStore:
    """Stores embeddings inside a FAISS index."""

    def __init__(
        self,
        dimension: int,
    ) -> None:
        self.index = faiss.IndexFlatIP(dimension)
        self.metadata_store = MetadataStore()

    def add(
        self,
        embedding: list[float],
        chunk: DocumentChunk,
    ) -> None:

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

            results.append(
                SearchResult(
                    chunk=self.metadata_store.get(idx),
                    score=float(score),
                )
            )

        return results
