"""Validate and sanitise LLM-generated JSON for self-query filters."""

from __future__ import annotations

import json
import logging
from typing import Any

from app.retrieval.self_query.models import _SUPPORTED_FIELDS, StructuredQuery

logger = logging.getLogger(__name__)


class FilterValidator:
    """Parse raw LLM output into a validated ``StructuredQuery``.

    Responsibilities
    ----------------
    1. Parse the LLM response as JSON.
    2. Drop any filter keys not in the supported set.
    3. Coerce types (everything is stored as ``str | None``).
    4. Return a ``StructuredQuery``, or raise ``ValueError``.
    """

    def validate(self, raw: str, question: str) -> StructuredQuery:
        """Validate *raw* JSON and return a ``StructuredQuery``.

        Parameters
        ----------
        raw:
            The raw string returned by the LLM.
        question:
            The original user question (used as fallback
            ``rewritten_query`` when parsing fails).

        Returns
        -------
        StructuredQuery

        Raises
        ------
        ValueError
            If *raw* is not valid JSON or doesn't contain the required
            fields.
        """
        # 1. Parse JSON ---------------------------------------------------
        try:
            data: dict[str, Any] = json.loads(raw.strip())
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON from LLM: {exc}") from exc

        # 2. Validate structure -------------------------------------------
        if not isinstance(data, dict):
            raise ValueError(f"Expected JSON object, got {type(data).__name__}")

        rewritten = data.get("rewritten_query", "")
        if not isinstance(rewritten, str) or not rewritten.strip():
            raise ValueError("Missing or empty 'rewritten_query' field")

        raw_filters = data.get("metadata_filters", {})
        if not isinstance(raw_filters, dict):
            raise ValueError("'metadata_filters' must be a JSON object")

        # 3. Build StructuredQuery ----------------------------------------
        clean_filters: dict[str, str | None] = {}
        for key, value in raw_filters.items():
            if key not in _SUPPORTED_FIELDS:
                logger.debug("Dropping unsupported filter key: %s", key)
                continue
            if value is not None:
                clean_filters[key] = str(value)
            else:
                clean_filters[key] = None

        return StructuredQuery(
            rewritten_query=rewritten.strip(),
            metadata_filters=clean_filters,
        )


# ---------------------------------------------------------------------------
# convenience helper
# ---------------------------------------------------------------------------


def apply_filters(
    results: list[Any],
    filters: dict[str, str | None],
) -> list[Any]:
    """Apply metadata filters to a list of ``SearchResult`` objects.

    Delegates to ``MetadataFilter.filter``, converting the flat
    ``{key: value}`` dict into the keyword-argument form it expects.
    """
    from app.retrieval.filters.metadata_filter import MetadataFilter

    if not filters or all(v is None for v in filters.values()):
        return results

    tags_raw = filters.get("tags")
    kwargs: dict[str, str | list[str] | None] = {
        "source": filters.get("source"),
        "document_type": filters.get("document_type"),
        "tags": (
            [t.strip() for t in tags_raw.split(",")] if tags_raw else None
        ),
    }
    filtered = MetadataFilter.filter(results, **kwargs)  # type: ignore[arg-type]

    logger.debug(
        "Metadata filter applied — before=%d  after=%d  filters=%s",
        len(results),
        len(filtered),
        {k: v for k, v in filters.items() if v is not None},
    )
    return filtered
