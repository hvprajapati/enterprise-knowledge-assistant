from app.ingestion.models import DocumentChunk


class MetadataStore:
    """Stores chunk metadata in memory."""

    def __init__(self) -> None:
        self._chunks: dict[int, DocumentChunk] = {}

    def add(
        self,
        index: int,
        chunk: DocumentChunk,
    ) -> None:
        self._chunks[index] = chunk

    def get(
        self,
        index: int,
    ) -> DocumentChunk:
        try:
            return self._chunks[index]
        except KeyError:
            raise KeyError(f"Chunk index {index} not found in metadata store") from None
