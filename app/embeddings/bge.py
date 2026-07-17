from sentence_transformers import SentenceTransformer

from app.embeddings.base import BaseEmbeddingModel


class BGEEmbeddingModel(BaseEmbeddingModel):
    """BAAI BGE embedding model."""

    def __init__(self) -> None:
        self.model = SentenceTransformer("BAAI/bge-base-en-v1.5")

    @property
    def dimension(self) -> int:
        dim = self.model.get_embedding_dimension()
        if dim is None:
            raise RuntimeError(
                "Could not determine embedding dimension for BAAI/bge-base-en-v1.5"
            )
        return dim

    def encode(self, text: str) -> list[float]:
        embedding = self.model.encode(
            text,
            normalize_embeddings=True,
        )

        return embedding.tolist()

    def encode_batch(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
        )

        return embeddings.tolist()
