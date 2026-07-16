from app.embeddings.service import EmbeddingService
from app.ingestion.models import SearchResult
from app.vectorstore.faiss_store import FAISSVectorStore


class Retriever:
    """Retrieves relevant document chunks."""

    def __init__(
        self,
        vector_store: FAISSVectorStore,
    ) -> None:
        self.vector_store = vector_store
        self.embedding_service = EmbeddingService()

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[SearchResult]:
        query_embedding = self.embedding_service.model.encode(query)

        return self.vector_store.search(
            embedding=query_embedding,
            top_k=top_k,
        )
