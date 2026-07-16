from app.ingestion.models import SearchResult


class MetadataFilter:
    """Filters retrieved search results using document metadata."""

    @staticmethod
    def filter(
        results: list[SearchResult],
        *,
        source: str | None = None,
        document_type: str | None = None,
        tags: list[str] | None = None,
    ) -> list[SearchResult]:

        filtered = results

        if source:
            filtered = [result for result in filtered if result.chunk.metadata.source == source]

        if document_type:
            filtered = [
                result
                for result in filtered
                if result.chunk.metadata.document_type == document_type
            ]

        if tags:
            filtered = [
                result for result in filtered if set(tags).issubset(set(result.chunk.metadata.tags))
            ]

        return filtered
