"""Pydantic models for self-query metadata extraction."""

from __future__ import annotations

from pydantic import BaseModel, Field

# Fields the LLM is allowed to populate.  Only keys listed here survive
# validation; everything else is silently dropped.
_SUPPORTED_FIELDS = frozenset(
    {"filename", "source", "document_type", "tags", "extension", "year", "author"}
)


class StructuredQuery(BaseModel):
    """A question decomposed into rewritten text + metadata filters.

    Attributes
    ----------
    rewritten_query:
        The content-only question with metadata constraints removed.
    metadata_filters:
        Extracted key-value pairs.  ``None`` means no constraint.
        Unknown keys are dropped during validation.
    """

    rewritten_query: str = Field(
        default="",
        min_length=1,
        description="Content-only question with metadata constraints removed.",
    )
    metadata_filters: dict[str, str | None] = Field(
        default_factory=dict,
        description="Metadata key-value pairs. None = no filter applied.",
    )

    @classmethod
    def empty(cls, question: str) -> StructuredQuery:
        """Return a query with no filters (fallback on parse failure)."""
        return cls(rewritten_query=question, metadata_filters={})
