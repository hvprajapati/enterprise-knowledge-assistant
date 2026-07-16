from sentence_transformers import SentenceTransformer

from app.embeddings.base import BaseEmbeddingModel


class BGEEmbeddingModel(BaseEmbeddingModel):
    """BAAI BGE embedding model."""

    def __init__(self) -> None:
        self.model = SentenceTransformer("BAAI/bge-base-en-v1.5")

    def encode(self, text: str) -> list[float]:
        embedding = self.model.encode(
            text,
            normalize_embeddings=True,
        )

        return embedding.tolist()  # type: ignore[no-any-return]

    def encode_batch(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
        )

        return embeddings.tolist()  # type: ignore[no-any-return]
