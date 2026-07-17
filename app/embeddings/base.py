from abc import ABC, abstractmethod


class BaseEmbeddingModel(ABC):
    """Interface for embedding models."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Embedding vector dimension."""
        raise NotImplementedError

    @abstractmethod
    def encode(
        self,
        text: str,
    ) -> list[float]:
        raise NotImplementedError

    @abstractmethod
    def encode_batch(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        raise NotImplementedError