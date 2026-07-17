from app.embeddings.bge import BGEEmbeddingModel
from app.ingestion.models import DocumentChunk


class EmbeddingService:
    """Generates embeddings for document chunks and raw queries."""

    def __init__(self) -> None:
        self.model = BGEEmbeddingModel()

    @property
    def dimension(self) -> int:
        return self.model.dimension

    def embed_query(self, text: str) -> list[float]:
        """Encode a raw user query string.

        Returns
        -------
        list[float]
            Normalised embedding vector whose length equals ``dimension``.
        """
        return self.model.encode(text)

    def embed_chunk(
        self,
        chunk: DocumentChunk,
    ) -> list[float]:
        return self.model.encode(chunk.text)

    def embed_queries(
        self,
        queries: list[str],
    ) -> list[list[float]]:
        """Encode multiple raw query strings in a single batch.

        Returns
        -------
        list[list[float]]
            One embedding vector per input query.
        """
        return self.model.encode_batch(queries)

    def embed_chunks(
        self,
        chunks: list[DocumentChunk],
    ) -> list[list[float]]:
        texts = [chunk.text for chunk in chunks]
        return self.model.encode_batch(texts)