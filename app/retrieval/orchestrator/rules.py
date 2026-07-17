"""Deterministic classification rules for question analysis.

Each rule is a ``(predicate, QuestionType)`` pair evaluated in order.
The first matching rule wins.  Rules are intentionally simple and
deterministic so they can be replaced with an LLM classifier later
without changing the orchestrator interface.
"""

from __future__ import annotations

import re

from app.retrieval.orchestrator.models import QuestionType

# ---------------------------------------------------------------------------
# keyword / pattern sets
# ---------------------------------------------------------------------------

_COMPARISON_WORDS = {
    "vs", "versus", "compare", "comparison", "difference between",
    "diff between", "better", "worse", "pros and cons",
}

_METADATA_PATTERNS = [
    r"\b(?:show|find|search|list|give)\s+me\b",
    r"\bfrom\s+(?:\d{4})\b",          # "from 2024"
    r"\b(?:in|from)\s+the\s+(?:year|month)\b",
    r"\bby\s+(?:author|writer)\b",
    r"\.pdf\b|\.docx?\b|\.txt\b|\.md\b",
]

_TROUBLESHOOTING_WORDS = {
    "error", "bug", "fix", "issue", "problem", "fail", "crash",
    "broken", "wrong", "not working", "debug", "troubleshoot",
}

_AMBIGUOUS_PATTERNS = [
    r"\bhow does (?:it|this|that) work\b",
    r"\bwhat (?:is|are) (?:it|this|that)\b",
    r"\bexplain (?:it|this|that)\b",
    r"\btell me about (?:it|this|that)\b",
]

_BROAD_PATTERNS = [
    r"\bhow (?:does|do|can|should|would|might|could)\b",
    r"\bwhy (?:is|are|does|do)\b",
    r"\bwhat (?:are|is) the (?:benefits|advantages|disadvantages|pros|cons|implications)\b",
    r"\boverview\b|\bintroduction to\b|\bguide to\b",
    r"\bexplain\b.+\b(?:architecture|system|pipeline|framework|process)\b",
]


# ---------------------------------------------------------------------------
# classifier
# ---------------------------------------------------------------------------


def classify(question: str) -> QuestionType:
    """Return the best-fit ``QuestionType`` for *question*.

    Rules are evaluated in priority order.  The first matching rule
    wins; if no rule matches the question is classified as ``FACTUAL``.
    """
    text = question.lower().strip()

    # 1.  Comparison  (highest priority — explicit intent)
    if _match_any_word(text, _COMPARISON_WORDS):
        return QuestionType.COMPARISON

    # 2.  Troubleshooting
    if _match_any_word(text, _TROUBLESHOOTING_WORDS):
        return QuestionType.TROUBLESHOOTING

    # 3.  Metadata-aware  (file paths, dates, structured requests)
    if _match_any_pattern(text, _METADATA_PATTERNS):
        return QuestionType.METADATA

    # 4.  Ambiguous  (pronouns, vague references)
    if _match_any_pattern(text, _AMBIGUOUS_PATTERNS):
        return QuestionType.AMBIGUOUS

    # 5.  Broad  (how/why/what-benefits questions)
    if _match_any_pattern(text, _BROAD_PATTERNS):
        return QuestionType.BROAD

    # 6.  Keyword-heavy  (short, contains technical abbreviations)
    if _is_keyword_heavy(question):     # use original-case text
        return QuestionType.KEYWORD

    # 7.  Simple  (short, single-clause)
    if _is_simple(text):
        return QuestionType.SIMPLE

    # 8.  Default
    return QuestionType.FACTUAL


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _match_any_word(text: str, words: set[str]) -> bool:
    return any(w in text for w in words)


def _match_any_pattern(text: str, patterns: list[str]) -> bool:
    return any(re.search(p, text) for p in patterns)


def _is_keyword_heavy(text: str) -> bool:
    """Short question with technical acronyms / jargon.

    Checks the **original** text (before lowercasing) for uppercase
    abbreviations like GPU, RAM, CPU, FAISS, BM25.
    """
    words = text.split()
    if len(words) > 12:
        return False
    # Detect acronyms in the original-case text
    acronyms = re.findall(r"\b[A-Z]{2,}\b", text)
    return len(acronyms) >= 1 and len(words) <= 8


def _is_simple(text: str) -> bool:
    """Single-clause question with no complexity markers."""
    words = text.split()
    if len(words) > 10:
        return False
    # Must start with a WH-word or "explain"/"define"
    if not re.match(
        r"^(what|who|where|when|how|why|is|are|do|does|can|explain|define|list)\b",
        text,
    ):
        return False
    return True
