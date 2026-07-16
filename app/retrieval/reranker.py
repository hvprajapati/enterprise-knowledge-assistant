from sentence_transformers import CrossEncoder

from app.ingestion.models import SearchResult


class CrossEncoderReranker:
    """Reranks retrieved search results using a cross encoder."""

    def __init__(self) -> None:
        self.model = CrossEncoder("BAAI/bge-reranker-base")

    def rerank(
        self,
        query: str,
        results: list[SearchResult],
        top_k: int = 5,
    ) -> list[SearchResult]:

        if not results:
            return []

        pairs = [(query, result.chunk.text) for result in results]

        scores = self.model.predict(pairs)

        reranked = []

        for result, score in zip(results, scores, strict=True):
            result.score = float(score)
            reranked.append(result)

        reranked.sort(
            key=lambda result: result.score,
            reverse=True,
        )

        return reranked[:top_k]
