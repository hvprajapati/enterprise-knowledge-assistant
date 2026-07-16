from abc import ABC, abstractmethod


class BaseEmbeddingModel(ABC):
    """Interface for embedding models."""

    @abstractmethod
    def encode(self, text: str) -> list[float]:
        raise NotImplementedError

    @abstractmethod
    def encode_batch(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        raise NotImplementedError
