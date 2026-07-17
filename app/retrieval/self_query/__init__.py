"""Self-Query Retrieval package.

Extracts structured metadata filters from natural-language questions.
"""

from app.retrieval.self_query.models import StructuredQuery
from app.retrieval.self_query.parser import SelfQueryParser
from app.retrieval.self_query.validator import FilterValidator

__all__ = [
    "FilterValidator",
    "SelfQueryParser",
    "StructuredQuery",
    "apply_filters",
]

# ---------------------------------------------------------------------------
# convenience re-export — used by hybrid.py
# ---------------------------------------------------------------------------

from app.retrieval.self_query.validator import apply_filters  # noqa: E402
