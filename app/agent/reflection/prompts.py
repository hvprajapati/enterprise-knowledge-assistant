"""Prompt templates for the Reflection Engine.

The reflection prompt is designed to produce a **structured JSON
evaluation** of an answer.  It intentionally does NOT ask the LLM
to rewrite, fix, or improve the answer — that is the job of
downstream nodes (Validation, Retry).

Design principles
-----------------
1. **Evaluate, don't modify.**  The prompt never asks for a corrected
   answer — only a verdict.
2. **Evidence-first.**  Every evaluation dimension references the
   retrieved context, not the model's internal knowledge.
3. **Structured output.**  The prompt asks for strict JSON matching
   the ``ReflectionResult`` schema so Pydantic validation can catch
   malformed outputs.
4. **Defensive.**  The prompt handles edge cases: empty context,
   "not found" answers, and conversational turns.
"""

from __future__ import annotations

REFLECTION_SYSTEM_PROMPT: str = """\
You are an expert evaluator for a Retrieval-Augmented Generation (RAG) system.
Your job is to inspect a generated answer and rate its quality.

RULES:
1. NEVER rewrite, correct, or improve the answer. Your output is a JSON evaluation ONLY.
2. Base your evaluation on two things:
   - Whether the answer fully addresses the user's question.
   - Whether the answer's claims are supported by the retrieved context.
3. Be strict but fair. Flag issues, but do not penalise the answer for
   limitations in the retrieved context (e.g. missing documents).
4. If the context is empty and the answer says "no documents found",
   that is CORRECT behaviour — rate it as grounded=unknown, complete=true.
5. If the answer contains factual claims NOT present in the context,
   flag grounded=ungrounded or grounded=partially_grounded.
6. If the answer is conversational (greeting, thanks),
   rate quality=excellent, grounded=unknown, complete=true.
7. Always output valid JSON matching the schema below."""

REFLECTION_USER_PROMPT: str = """\
Evaluate the following RAG answer.

=== USER QUESTION ===
{question}

=== RETRIEVED CONTEXT ===
{context}

=== GENERATED ANSWER ===
{answer}

=== EVALUATION DIMENSIONS ===

1. **Relevance**: Does the answer actually address the question? Or does it go off-topic?

2. **Groundedness**: Is every important factual claim in the answer supported by
   at least one passage in the retrieved context?  If the answer introduces
   facts not found in the context, it is partially or fully ungrounded.

3. **Completeness**: Does the answer fully answer the question, or is important
   information missing?  If the question asks for multiple things, does the
   answer address all of them?

4. **Specificity**: Is the answer detailed and specific, or is it overly vague?
   Vague answers should be flagged (e.g. "It depends on several factors" with
   no further detail).

5. **Confidence**: On a scale of 0.0 (terrible) to 1.0 (perfect), how confident
   are you that this is a satisfactory answer?

=== OUTPUT FORMAT ===

Return ONLY a single JSON object with exactly these fields:

{{
  "answer_quality": "excellent" | "good" | "adequate" | "inadequate" | "irrelevant",
  "grounded": "fully_grounded" | "partially_grounded" | "ungrounded" | "unknown",
  "complete": true | false,
  "relevant": true | false,
  "confidence_score": 0.0 to 1.0,
  "missing_information": ["topic A", "topic B"],
  "recommendations": ["suggestion 1", "suggestion 2"],
  "reasoning": "Brief explanation of your evaluation (2-4 sentences)."
}}

No preamble, no markdown fences, no extra text — ONLY the JSON object."""


# ---------------------------------------------------------------------------
# context formatting helper
# ---------------------------------------------------------------------------


def format_context_for_reflection(
    search_results: list[dict[str, object]],
    max_chunks: int = 10,
) -> str:
    """Format retrieved chunks as a compact text block for the evaluator.

    Parameters
    ----------
    search_results:
        Serialised search results from ``AgentState.search_results``.
    max_chunks:
        Maximum number of chunks to include (avoids blowing up the
        prompt).  Defaults to 10 — enough for grounding checks without
        excessive token usage.

    Returns
    -------
    str
        Formatted context block, or a placeholder if empty.
    """
    if not search_results:
        return "(No documents were retrieved — context is empty.)"

    lines: list[str] = []
    for i, r in enumerate(search_results[:max_chunks], start=1):
        chunk_id = r.get("chunk_id", "?")
        text = str(r.get("text", ""))
        filename = r.get("filename", "unknown")
        page = r.get("page", "?")
        score = r.get("score", 0.0)

        lines.append(
            f"[{i}] chunk_id={chunk_id}  file={filename}  page={page}  score={score:.4f}\n"
            f"    {text}"
        )

    header = (
        f"Showing {len(search_results[:max_chunks])} of "
        f"{len(search_results)} retrieved chunks:\n\n"
    )
    return header + "\n\n".join(lines)
