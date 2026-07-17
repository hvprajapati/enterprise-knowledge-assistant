"""Reflection Engine — evaluates answers without modifying them.

The ``ReflectionEngine`` is the core of the reflection node.  It takes
a question, an answer, and retrieved context, and returns a structured
``ReflectionResult``.

Design
------
- **Read-only.**  The engine never alters the answer — it only inspects.
- **LLM-powered.**  Uses the configured LLM with structured JSON output.
- **Pydantic-validated.**  The LLM's JSON response is parsed and validated
  so malformed outputs are caught immediately.
- **Graceful degradation.**  If the LLM call fails (network, timeout,
  malformed JSON), the engine returns a neutral ``ReflectionResult``
  that does not block the graph.

Why not rule-based?
    Groundedness and completeness checks require semantic understanding
    of both the answer and the retrieved context.  Simple heuristics
    (length check, keyword overlap) are too brittle for production use.
    An LLM-based evaluator catches hallucinations, vagueness, and
    off-topic answers that rules would miss.
"""

from __future__ import annotations

import json
import logging
import re
import time

from app.agent.reflection.models import (
    AnswerQuality,
    GroundedStatus,
    ReflectionResult,
)
from app.agent.reflection.prompts import (
    REFLECTION_SYSTEM_PROMPT,
    REFLECTION_USER_PROMPT,
    format_context_for_reflection,
)
from app.llm.base import BaseLLM

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# patterns for cleaning LLM JSON output
# ---------------------------------------------------------------------------

_JSON_FENCE_RE = re.compile(r"^```(?:json)?\s*$", re.MULTILINE)
_JSON_MARKER_RE = re.compile(r"\{[\s\S]*\}")


class ReflectionEngine:
    """Evaluate a generated answer using the configured LLM.

    Usage::

        engine = ReflectionEngine(llm=my_llm)
        result = engine.reflect(
            question="What is FAISS?",
            answer="FAISS is a library for efficient similarity search...",
            retrieved_chunks=[...],
        )
        print(result.confidence_score)
    """

    def __init__(self, *, llm: BaseLLM) -> None:
        self._llm = llm

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def reflect(
        self,
        question: str,
        answer: str,
        retrieved_chunks: list[dict[str, object]],
    ) -> ReflectionResult:
        """Evaluate *answer* against *question* and *retrieved_chunks*.

        Parameters
        ----------
        question:
            The original (or rewritten) user question.
        answer:
            The answer produced by ``generate_node``.
        retrieved_chunks:
            Serialised search results from ``AgentState.search_results``.

        Returns
        -------
        ReflectionResult
            Always returns a result — on failure returns a neutral
            default so the graph never halts.
        """
        t_start = time.monotonic()
        logger.info(
            "Reflection started — question=%d chars  answer=%d chars  chunks=%d",
            len(question),
            len(answer),
            len(retrieved_chunks),
        )

        # Quick short-circuit: empty answer
        if not answer or not answer.strip():
            return self._empty_answer_result(question, t_start)

        try:
            # Build the context block
            context_block = format_context_for_reflection(retrieved_chunks)

            # Build the user prompt
            user_prompt = REFLECTION_USER_PROMPT.format(
                question=question,
                context=context_block,
                answer=answer,
            )

            # Call the LLM
            raw = self._llm.generate(
                prompt=user_prompt,
                system_prompt=REFLECTION_SYSTEM_PROMPT,
                temperature=0.0,
                max_tokens=1024,
            )

            # Parse & validate
            result = self._parse_response(raw)
            self._log_result(result, t_start)
            return result

        except Exception as exc:
            logger.exception("Reflection engine failed — returning default result")
            return self._fallback_result(question, answer, str(exc), t_start)

    # ------------------------------------------------------------------
    # response parsing
    # ------------------------------------------------------------------

    def _parse_response(self, raw: str) -> ReflectionResult:
        """Extract JSON from the LLM response and validate.

        The LLM may wrap the JSON in markdown fences or include a
        preamble.  We attempt to extract the first JSON object found,
        then parse and validate.
        """
        # Strip markdown fences if present
        cleaned = _JSON_FENCE_RE.sub("", raw).strip()

        # Find the first JSON object
        match = _JSON_MARKER_RE.search(cleaned)
        if not match:
            raise ValueError(f"No JSON object found in LLM response: {raw[:200]}")

        data = json.loads(match.group(0))

        # Normalise string enums to lowercase (LLMs sometimes capitalise)
        quality = str(data.get("answer_quality", "adequate")).lower()
        grounded = str(data.get("grounded", "unknown")).lower()

        return ReflectionResult(
            answer_quality=AnswerQuality(quality),
            grounded=GroundedStatus(grounded),
            complete=bool(data.get("complete", False)),
            relevant=bool(data.get("relevant", True)),
            confidence_score=float(data.get("confidence_score", 0.5)),
            missing_information=self._ensure_str_list(
                data.get("missing_information", [])
            ),
            recommendations=self._ensure_str_list(
                data.get("recommendations", [])
            ),
            reasoning=str(data.get("reasoning", "")),
        )

    # ------------------------------------------------------------------
    # fallback / edge-case helpers
    # ------------------------------------------------------------------

    def _empty_answer_result(
        self,
        question: str,
        t_start: float,
    ) -> ReflectionResult:
        """Handle the edge case where generate_node produced no answer."""
        elapsed = (time.monotonic() - t_start) * 1000
        result = ReflectionResult(
            answer_quality=AnswerQuality.INADEQUATE,
            grounded=GroundedStatus.UNKNOWN,
            complete=False,
            relevant=False,
            confidence_score=0.0,
            missing_information=["Answer was empty or whitespace-only."],
            recommendations=["Regenerate — the generate node produced an empty answer."],
            reasoning="The answer is empty. This is a generation failure.",
        )
        logger.warning("Reflection — empty answer detected  latency=%.0fms", elapsed)
        return result

    def _fallback_result(
        self,
        question: str,
        answer: str,
        error: str,
        t_start: float,
    ) -> ReflectionResult:
        """Return a neutral default when the engine itself fails."""
        elapsed = (time.monotonic() - t_start) * 1000
        result = ReflectionResult.default_result(
            question=question,
            answer=answer,
            error=error,
        )
        logger.warning(
            "Reflection — fallback used  latency=%.0fms  error=%s",
            elapsed,
            error[:120],
        )
        return result

    def _log_result(
        self,
        result: ReflectionResult,
        t_start: float,
    ) -> None:
        """Emit structured log line for observability."""
        elapsed = (time.monotonic() - t_start) * 1000
        log_dict = result.to_log_dict()
        logger.info(
            "Reflection complete — "
            "quality=%s  grounded=%s  complete=%s  relevant=%s  "
            "confidence=%.2f  missing=%d  recs=%d  latency=%.0fms",
            log_dict["quality"],
            log_dict["grounded"],
            log_dict["complete"],
            log_dict["relevant"],
            log_dict["confidence"],
            log_dict["missing_count"],
            log_dict["rec_count"],
            elapsed,
        )

    @staticmethod
    def _ensure_str_list(value: object) -> list[str]:
        """Coerce a value from JSON into a list of strings."""
        if isinstance(value, list):
            return [str(item) for item in value]
        if isinstance(value, str):
            return [value] if value else []
        return []
