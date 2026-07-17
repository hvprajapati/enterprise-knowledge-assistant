from app.embeddings.bge import BGEEmbeddingModel
from app.ingestion.models import DocumentChunk


class EmbeddingService:
    """Generates embeddings for document chunks."""

    def __init__(self) -> None:
        self.model = BGEEmbeddingModel()

    @property
    def dimension(self) -> int:
        return self.model.dimension

    def embed_chunk(
        self,
        chunk: DocumentChunk,
    ) -> list[float]:
        return self.model.encode(chunk.text)

    def embed_chunks(
        self,
        chunks: list[DocumentChunk],
    ) -> list[list[float]]:
        texts = [
            chunk.text
            for chunk in chunks
        ]

        return self.model.encode_batch(texts)