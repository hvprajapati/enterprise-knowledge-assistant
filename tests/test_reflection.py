"""Unit tests for the Reflection Engine and Reflection Node.

Covers:
- high-quality answer evaluation
- incomplete answer detection
- hallucinated (ungrounded) answer detection
- irrelevant answer detection
- reflection engine failure / fallback
- empty answer edge case
- ReflectionResult model validation
- format_context_for_reflection helper
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from app.agent.reflection import (
    AnswerQuality,
    GroundedStatus,
    ReflectionEngine,
    ReflectionResult,
)
from app.agent.reflection.prompts import format_context_for_reflection
from app.agent.reflection.reflection import (
    REFLECTION_SYSTEM_PROMPT,
    REFLECTION_USER_PROMPT,
)

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _make_chunks(
    chunks: list[dict[str, object]] | None = None,
) -> list[dict[str, object]]:
    """Return a reasonable set of serialised search results for testing."""
    if chunks is not None:
        return chunks
    return [
        {
            "chunk_id": "c1",
            "text": (
                "FAISS is a library for efficient similarity search "
                "and clustering of dense vectors."
            ),
            "score": 0.95,
            "filename": "faiss.pdf",
            "page": 1,
        },
        {
            "chunk_id": "c2",
            "text": (
                "FAISS supports GPU acceleration and multiple index "
                "types including IndexFlatL2 and IndexIVFFlat."
            ),
            "score": 0.87,
            "filename": "faiss.pdf",
            "page": 2,
        },
        {
            "chunk_id": "c3",
            "text": (
                "BM25 is a bag-of-words retrieval function that "
                "ranks documents by term frequency."
            ),
            "score": 0.45,
            "filename": "retrieval.pdf",
            "page": 5,
        },
    ]


def _build_reflection_json(
    *,
    answer_quality: str = "good",
    grounded: str = "fully_grounded",
    complete: bool = True,
    relevant: bool = True,
    confidence_score: float = 0.85,
    missing_information: list[str] | None = None,
    recommendations: list[str] | None = None,
    reasoning: str = "The answer is accurate and well-supported.",
) -> str:
    """Build a valid LLM-like reflection JSON response."""
    return json.dumps({
        "answer_quality": answer_quality,
        "grounded": grounded,
        "complete": complete,
        "relevant": relevant,
        "confidence_score": confidence_score,
        "missing_information": missing_information or [],
        "recommendations": recommendations or [],
        "reasoning": reasoning,
    })


# ---------------------------------------------------------------------------
# ReflectionResult model tests
# ---------------------------------------------------------------------------


class TestReflectionResult:
    """Pydantic model validation and factory tests."""

    def test_default_construction(self) -> None:
        result = ReflectionResult()
        assert result.answer_quality == AnswerQuality.ADEQUATE
        assert result.grounded == GroundedStatus.UNKNOWN
        assert result.complete is False
        assert result.relevant is True
        assert result.confidence_score == 0.5
        assert result.missing_information == []
        assert result.recommendations == []
        assert result.reasoning == ""

    def test_full_construction(self) -> None:
        result = ReflectionResult(
            answer_quality=AnswerQuality.EXCELLENT,
            grounded=GroundedStatus.FULLY_GROUNDED,
            complete=True,
            relevant=True,
            confidence_score=0.95,
            missing_information=[],
            recommendations=["None needed."],
            reasoning="Perfect answer.",
        )
        assert result.answer_quality == AnswerQuality.EXCELLENT
        assert result.confidence_score == 0.95

    def test_confidence_score_bounds(self) -> None:
        # Below 0.0 — should fail
        with pytest.raises(ValidationError):
            ReflectionResult(confidence_score=-0.1)
        # Above 1.0 — should fail
        with pytest.raises(ValidationError):
            ReflectionResult(confidence_score=1.1)

    def test_default_result_factory(self) -> None:
        result = ReflectionResult.default_result(
            question="What is FAISS?",
            answer="FAISS is a library.",
            error="LLM timeout",
        )
        assert result.answer_quality == AnswerQuality.ADEQUATE
        assert result.grounded == GroundedStatus.UNKNOWN
        assert result.complete is False
        assert result.confidence_score == 0.5
        assert any("manual review" in r for r in result.recommendations)

    def test_to_log_dict(self) -> None:
        result = ReflectionResult(
            answer_quality=AnswerQuality.GOOD,
            grounded=GroundedStatus.FULLY_GROUNDED,
            complete=True,
            relevant=True,
            confidence_score=0.88,
            missing_information=["topic X"],
            recommendations=["use broader terms"],
            reasoning="Solid.",
        )
        log = result.to_log_dict()
        assert log == {
            "quality": "good",
            "grounded": "fully_grounded",
            "complete": True,
            "relevant": True,
            "confidence": 0.88,
            "missing_count": 1,
            "rec_count": 1,
        }

    def test_model_dump_for_graph(self) -> None:
        """model_dump() should produce a JSON-serialisable dict for AgentState."""
        result = ReflectionResult(
            answer_quality=AnswerQuality.ADEQUATE,
            grounded=GroundedStatus.PARTIALLY_GROUNDED,
            reasoning="Some claims are unsupported.",
        )
        dumped = result.model_dump()
        assert isinstance(dumped, dict)
        assert dumped["answer_quality"] == "adequate"
        assert dumped["grounded"] == "partially_grounded"
        # Round-trip through JSON
        assert json.loads(json.dumps(dumped)) == dumped


# ---------------------------------------------------------------------------
# AnswerQuality / GroundedStatus enum tests
# ---------------------------------------------------------------------------


class TestEnums:
    def test_answer_quality_values(self) -> None:
        assert set(AnswerQuality) == {
            AnswerQuality.EXCELLENT,
            AnswerQuality.GOOD,
            AnswerQuality.ADEQUATE,
            AnswerQuality.INADEQUATE,
            AnswerQuality.IRRELEVANT,
        }

    def test_grounded_status_values(self) -> None:
        assert set(GroundedStatus) == {
            GroundedStatus.FULLY_GROUNDED,
            GroundedStatus.PARTIALLY_GROUNDED,
            GroundedStatus.UNGROUNDED,
            GroundedStatus.UNKNOWN,
        }

    def test_str_enum_comparison(self) -> None:
        assert AnswerQuality("good") == AnswerQuality.GOOD
        assert GroundedStatus("fully_grounded") == GroundedStatus.FULLY_GROUNDED


# ---------------------------------------------------------------------------
# format_context_for_reflection tests
# ---------------------------------------------------------------------------


class TestFormatContext:
    def test_empty_context(self) -> None:
        result = format_context_for_reflection([])
        assert "empty" in result.lower()

    def test_max_chunks_respected(self) -> None:
        chunks = [
            {
                "chunk_id": f"c{i}",
                "text": f"Chunk {i}",
                "filename": "doc.pdf",
                "page": i,
                "score": 0.9,
            }
            for i in range(20)
        ]
        result = format_context_for_reflection(chunks, max_chunks=5)
        # Should only include 5 chunks (5 "[N]" markers)
        assert result.count("[") == 5
        assert "5 of 20" in result

    def test_includes_metadata_fields(self) -> None:
        chunks = _make_chunks()
        result = format_context_for_reflection(chunks)
        assert "chunk_id=c1" in result
        assert "file=faiss.pdf" in result
        assert "score=0.9500" in result


# ---------------------------------------------------------------------------
# ReflectionEngine tests (with mocked LLM)
# ---------------------------------------------------------------------------


class TestReflectionEngine:
    """Tests for ReflectionEngine.reflect() with a mocked LLM."""

    def test_high_quality_answer(self) -> None:
        """An excellent, well-grounded answer."""
        mock_llm = MagicMock()
        mock_llm.generate.return_value = _build_reflection_json(
            answer_quality="excellent",
            grounded="fully_grounded",
            complete=True,
            relevant=True,
            confidence_score=0.95,
            reasoning="Clear, accurate, fully supported by context.",
        )

        engine = ReflectionEngine(llm=mock_llm)
        result = engine.reflect(
            question="What is FAISS?",
            answer=(
                "FAISS is a library for efficient similarity search "
                "and clustering of dense vectors. It supports GPU acceleration."
            ),
            retrieved_chunks=_make_chunks(),
        )

        assert result.answer_quality == AnswerQuality.EXCELLENT
        assert result.grounded == GroundedStatus.FULLY_GROUNDED
        assert result.complete is True
        assert result.relevant is True
        assert result.confidence_score == 0.95
        mock_llm.generate.assert_called_once()

    def test_incomplete_answer(self) -> None:
        """An answer that is partially correct but missing key information."""
        mock_llm = MagicMock()
        mock_llm.generate.return_value = _build_reflection_json(
            answer_quality="adequate",
            grounded="partially_grounded",
            complete=False,
            relevant=True,
            confidence_score=0.55,
            missing_information=["GPU support details", "Index types available in FAISS"],
            recommendations=["Expand answer to cover FAISS index types and GPU support."],
            reasoning="The answer covers the basics but omits important features.",
        )

        engine = ReflectionEngine(llm=mock_llm)
        result = engine.reflect(
            question="What is FAISS and what features does it offer?",
            answer="FAISS is a library for similarity search.",
            retrieved_chunks=_make_chunks(),
        )

        assert result.answer_quality == AnswerQuality.ADEQUATE
        assert result.grounded == GroundedStatus.PARTIALLY_GROUNDED
        assert result.complete is False
        assert len(result.missing_information) == 2

    def test_hallucinated_answer(self) -> None:
        """Answer contains claims not found in retrieved context."""
        mock_llm = MagicMock()
        mock_llm.generate.return_value = _build_reflection_json(
            answer_quality="inadequate",
            grounded="ungrounded",
            complete=False,
            relevant=True,
            confidence_score=0.2,
            missing_information=["Accurate FAISS description"],
            recommendations=[
                "Rewrite answer using ONLY retrieved context.",
                "Remove claim about FAISS being a database — it is a library.",
            ],
            reasoning=(
                "The answer calls FAISS a 'database' — "
                "this is false and unsupported by context."
            ),
        )

        engine = ReflectionEngine(llm=mock_llm)
        result = engine.reflect(
            question="What is FAISS?",
            answer="FAISS is a vector database developed by Google for storing images.",
            retrieved_chunks=_make_chunks(),
        )

        assert result.answer_quality == AnswerQuality.INADEQUATE
        assert result.grounded == GroundedStatus.UNGROUNDED
        assert result.confidence_score == 0.2

    def test_irrelevant_answer(self) -> None:
        """Answer completely off-topic."""
        mock_llm = MagicMock()
        mock_llm.generate.return_value = _build_reflection_json(
            answer_quality="irrelevant",
            grounded="unknown",
            complete=False,
            relevant=False,
            confidence_score=0.05,
            missing_information=["Any information about FAISS"],
            recommendations=[
                "The answer discusses SQL databases — not the question about FAISS.",
                "Regenerate with the correct topic.",
            ],
            reasoning="Answer is about SQL databases; the question was about FAISS vector search.",
        )

        engine = ReflectionEngine(llm=mock_llm)
        result = engine.reflect(
            question="What is FAISS?",
            answer=(
                "SQL databases use tables, rows, and columns to store "
                "structured data. They support ACID transactions."
            ),
            retrieved_chunks=_make_chunks(),
        )

        assert result.answer_quality == AnswerQuality.IRRELEVANT
        assert result.relevant is False
        assert result.confidence_score <= 0.1  # very low

    def test_empty_answer(self) -> None:
        """Empty or whitespace-only answer should short-circuit."""
        mock_llm = MagicMock()
        engine = ReflectionEngine(llm=mock_llm)

        result = engine.reflect(
            question="What is FAISS?",
            answer="   ",
            retrieved_chunks=_make_chunks(),
        )

        assert result.answer_quality == AnswerQuality.INADEQUATE
        assert result.complete is False
        assert result.confidence_score == 0.0
        # LLM should NOT have been called
        mock_llm.generate.assert_not_called()

    def test_llm_failure_returns_fallback(self) -> None:
        """When the LLM call raises, return a neutral fallback."""
        mock_llm = MagicMock()
        mock_llm.generate.side_effect = RuntimeError("Connection refused")

        engine = ReflectionEngine(llm=mock_llm)
        result = engine.reflect(
            question="What is FAISS?",
            answer="FAISS is a library.",
            retrieved_chunks=_make_chunks(),
        )

        assert result.answer_quality == AnswerQuality.ADEQUATE
        assert result.grounded == GroundedStatus.UNKNOWN
        assert result.confidence_score == 0.5
        assert "Reflection engine failed" in result.reasoning

    def test_malformed_json_response(self) -> None:
        """LLM returns garbage that can't be parsed."""
        mock_llm = MagicMock()
        mock_llm.generate.return_value = "Here is my evaluation: blablabla not JSON"

        engine = ReflectionEngine(llm=mock_llm)
        result = engine.reflect(
            question="What is FAISS?",
            answer="FAISS is a library.",
            retrieved_chunks=_make_chunks(),
        )

        # Should fallback gracefully
        assert result.confidence_score == 0.5
        assert result.grounded == GroundedStatus.UNKNOWN

    def test_json_in_markdown_fence(self) -> None:
        """LLM wraps JSON in a markdown code fence — engine should strip it."""
        mock_llm = MagicMock()
        mock_llm.generate.return_value = "```json\n" + _build_reflection_json(
            answer_quality="good",
            grounded="fully_grounded",
        ) + "\n```"

        engine = ReflectionEngine(llm=mock_llm)
        result = engine.reflect(
            question="What is FAISS?",
            answer="FAISS is a library.",
            retrieved_chunks=_make_chunks(),
        )

        assert result.answer_quality == AnswerQuality.GOOD
        assert result.grounded == GroundedStatus.FULLY_GROUNDED

    def test_llm_capitalizes_enum_values(self) -> None:
        """LLM returns "Excellent" instead of "excellent" — should normalise."""
        mock_llm = MagicMock()
        mock_llm.generate.return_value = json.dumps({
            "answer_quality": "Excellent",
            "grounded": "Fully_Grounded",
            "complete": True,
            "relevant": True,
            "confidence_score": 0.9,
            "missing_information": [],
            "recommendations": [],
            "reasoning": "Good.",
        })

        engine = ReflectionEngine(llm=mock_llm)
        result = engine.reflect(
            question="What is FAISS?",
            answer="FAISS is a library.",
            retrieved_chunks=_make_chunks(),
        )

        assert result.answer_quality == AnswerQuality.EXCELLENT
        assert result.grounded == GroundedStatus.FULLY_GROUNDED

    def test_system_prompt_includes_no_rewrite_instruction(self) -> None:
        """The system prompt must NEVER instruct the LLM to rewrite the answer."""
        assert "NEVER rewrite" in REFLECTION_SYSTEM_PROMPT
        lower_prompt = REFLECTION_SYSTEM_PROMPT.lower()
        before_rewrite = lower_prompt.split("rewrite")[0]
        assert "correct" not in before_rewrite

    def test_conversational_answer(self) -> None:
        """Greeting/conversational answers should be rated excellent + unknown grounding."""
        mock_llm = MagicMock()
        mock_llm.generate.return_value = _build_reflection_json(
            answer_quality="excellent",
            grounded="unknown",
            complete=True,
            relevant=True,
            confidence_score=0.95,
            reasoning="Conversational greeting — no context needed.",
        )

        engine = ReflectionEngine(llm=mock_llm)
        result = engine.reflect(
            question="Hello!",
            answer="Hello! How can I help you today?",
            retrieved_chunks=[],
        )

        assert result.answer_quality == AnswerQuality.EXCELLENT
        assert result.grounded == GroundedStatus.UNKNOWN
        assert result.complete is True


# ---------------------------------------------------------------------------
# Prompt template tests
# ---------------------------------------------------------------------------


class TestPrompts:
    def test_user_prompt_includes_question_context_answer(self) -> None:
        prompt = REFLECTION_USER_PROMPT.format(
            question="Q",
            context="CTX",
            answer="A",
        )
        assert "Q" in prompt
        assert "CTX" in prompt
        assert "A" in prompt
        assert "=== USER QUESTION ===" in prompt
        assert "=== RETRIEVED CONTEXT ===" in prompt
        assert "=== GENERATED ANSWER ===" in prompt

    def test_user_prompt_includes_evaluation_dimensions(self) -> None:
        prompt = REFLECTION_USER_PROMPT.format(
            question="Q",
            context="CTX",
            answer="A",
        )
        assert "Relevance" in prompt
        assert "Groundedness" in prompt
        assert "Completeness" in prompt
        assert "Specificity" in prompt
        assert "Confidence" in prompt

    def test_user_prompt_asks_for_json_only(self) -> None:
        prompt = REFLECTION_USER_PROMPT.format(
            question="Q",
            context="CTX",
            answer="A",
        )
        assert "ONLY the JSON object" in prompt
        assert "No preamble" in prompt


# ---------------------------------------------------------------------------
# integration-style: reflection in the context of a graph node
# ---------------------------------------------------------------------------


class TestReflectionNodeIntegration:
    """Lightweight tests that exercise the node function directly."""

    def test_reflection_node_stores_result(self) -> None:
        """Simulate the reflection_node function with a mocked engine."""
        from app.agent.nodes import _services, reflection_node

        # Setup
        mock_llm = MagicMock()
        mock_llm.generate.return_value = _build_reflection_json(
            answer_quality="good",
            grounded="fully_grounded",
            complete=True,
            confidence_score=0.88,
            reasoning="Solid answer.",
        )

        engine = ReflectionEngine(llm=mock_llm)
        _services["reflection_engine"] = engine

        state: dict = {
            "question": "What is FAISS?",
            "rewritten_question": None,
            "search_results": _make_chunks(),
            "answer": "FAISS is a library for similarity search.",
            "executed_nodes": ["planner", "rewrite", "retrieve", "generate"],
        }

        result = reflection_node(state)  # type: ignore[arg-type]

        assert "reflection_result" in result
        rr = result["reflection_result"]
        assert rr["answer_quality"] == "good"
        assert rr["grounded"] == "fully_grounded"
        assert "reflection" in result["executed_nodes"]

    def test_reflection_node_with_rewritten_question(self) -> None:
        """Uses rewritten_question when available."""
        from app.agent.nodes import _services, reflection_node

        mock_llm = MagicMock()
        mock_llm.generate.return_value = _build_reflection_json(
            answer_quality="excellent",
            grounded="fully_grounded",
            complete=True,
            confidence_score=0.92,
        )

        engine = ReflectionEngine(llm=mock_llm)
        _services["reflection_engine"] = engine

        state: dict = {
            "question": "How does it work?",
            "rewritten_question": "How does FAISS perform efficient similarity search?",
            "search_results": _make_chunks(),
            "answer": "FAISS uses indexing structures like IVF for fast nearest-neighbor search.",
            "executed_nodes": ["planner", "rewrite", "retrieve", "generate"],
        }

        result = reflection_node(state)  # type: ignore[arg-type]

        # Verify the rewritten question was passed to the engine
        call_args = mock_llm.generate.call_args
        assert "How does FAISS perform" in str(call_args)
        assert result["reflection_result"]["answer_quality"] == "excellent"

    def test_reflection_node_engine_not_configured(self) -> None:
        """Raises RuntimeError if engine was never registered."""
        from app.agent.nodes import _services, reflection_node

        # Remove the engine
        _services.pop("reflection_engine", None)

        state: dict = {
            "question": "What is FAISS?",
            "search_results": _make_chunks(),
            "answer": "FAISS is a library.",
            "executed_nodes": ["generate"],
        }

        with pytest.raises(RuntimeError, match="reflection_engine"):
            reflection_node(state)  # type: ignore[arg-type]
