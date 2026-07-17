"""Deterministic question classifier for the agent planner.

Uses keyword/pattern rules.  Designed to be replaceable with an
LLM-based classifier later without changing the ``Planner`` interface.
"""

from __future__ import annotations

import re

from app.agent.planner.models import QuestionType

# ---------------------------------------------------------------------------
# patterns
# ---------------------------------------------------------------------------

_COMPARISON = {
    "vs", "versus", "compare", "comparison", "difference between",
    "better", "worse", "pros and cons",
}

_SUMMARIZE = {
    "summarize", "summarise", "summary", "tldr", "recap", "overview", "sum up",
}

_TROUBLESHOOT = {
    "error", "bug", "fix", "issue", "problem", "fail", "crash",
    "broken", "wrong", "not working", "debug",
}

_METADATA_PATTERNS = [
    r"\b(?:show|find|search|list|give)\s+me\b",
    r"\bfrom\s+(?:\d{4})\b",
    r"\bby\s+(?:author|writer)\b",
    r"\.pdf\b|\.docx?\b|\.txt\b|\.md\b",
    r"\b(?:filter|filtered)\s+by\b",
]

_BROAD_PATTERNS = [
    r"\bhow (?:does|do|can|should|would|might|could)\b",
    r"\bwhy (?:is|are|does|do)\b",
    r"\bwhat (?:are|is) the (?:benefits|advantages|disadvantages|pros|cons)\b",
    r"\bexplain\b.+\b(?:architecture|system|pipeline|framework|process)\b",
]

_CONVERSATIONAL = {"hello", "hi", "hey", "thanks", "thank you", "good morning", "goodbye", "help"}


def classify(question: str) -> QuestionType:
    """Return the best-fit ``QuestionType`` for *question*."""
    text = question.lower().strip()
    words = set(text.split())

    # 1. Conversational (greetings etc.)
    if words & _CONVERSATIONAL and len(words) <= 5:
        return QuestionType.CONVERSATIONAL

    # 2. Comparison
    if _any_word(text, _COMPARISON):
        return QuestionType.COMPARISON

    # 3. Troubleshooting
    if _any_word(text, _TROUBLESHOOT):
        return QuestionType.TROUBLESHOOTING

    # 4. Metadata lookup
    if _any_pattern(text, _METADATA_PATTERNS):
        return QuestionType.METADATA_LOOKUP

    # 5. Summarization
    if _any_word(text, _SUMMARIZE):
        return QuestionType.SUMMARIZATION

    # 6. Broad research
    if _any_pattern(text, _BROAD_PATTERNS) or len(text.split()) > 20:
        return QuestionType.BROAD_RESEARCH

    # 7. Factual (default)
    if re.match(r"^(what|who|where|when|how|why|is|are|do|does|can|define|list)\b", text):
        return QuestionType.FACTUAL

    return QuestionType.UNKNOWN


def _any_word(text: str, terms: set[str]) -> bool:
    return any(t in text for t in terms)


def _any_pattern(text: str, patterns: list[str]) -> bool:
    return any(re.search(p, text) for p in patterns)
