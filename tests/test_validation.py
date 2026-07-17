"""Unit tests for the Answer Validator and Validation Node.

Covers:
- valid answer (all checks pass)
- low confidence threshold failure
- hallucinated answer (not grounded)
- incomplete answer
- irrelevant answer
- validator failure / fallback
- threshold edge cases
- ValidationResult model serialization
- validation node integration
"""

from __future__ import annotations

import pytest

from app.agent.reflection.models import (
    AnswerQuality,
    GroundedStatus,
    ReflectionResult,
)
from app.agent.validation import (
    AnswerValidator,
    ValidationCheck,
    ValidationResult,
    ValidationSeverity,
    ValidationThresholds,
)

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _make_reflection(
    *,
    answer_quality: AnswerQuality = AnswerQuality.GOOD,
    grounded: GroundedStatus = GroundedStatus.FULLY_GROUNDED,
    complete: bool = True,
    relevant: bool = True,
    confidence_score: float = 0.85,
    missing_information: list[str] | None = None,
    recommendations: list[str] | None = None,
    reasoning: str = "Solid answer.",
) -> ReflectionResult:
    """Build a ReflectionResult with sensible defaults for testing."""
    return ReflectionResult(
        answer_quality=answer_quality,
        grounded=grounded,
        complete=complete,
        relevant=relevant,
        confidence_score=confidence_score,
        missing_information=missing_information or [],
        recommendations=recommendations or [],
        reasoning=reasoning,
    )


def _lenient_thresholds() -> ValidationThresholds:
    """Thresholds that almost everything passes."""
    return ValidationThresholds(
        min_confidence_score=0.3,
        require_grounded=False,
        require_completeness=False,
        require_relevance=False,
    )


def _strict_thresholds() -> ValidationThresholds:
    """Thresholds that require perfection."""
    return ValidationThresholds(
        min_confidence_score=0.95,
        require_grounded=True,
        require_completeness=True,
        require_relevance=True,
    )


def _default_thresholds() -> ValidationThresholds:
    """Production-like thresholds."""
    return ValidationThresholds(
        min_confidence_score=0.6,
        require_grounded=True,
        require_completeness=True,
        require_relevance=True,
    )


# ---------------------------------------------------------------------------
# ValidationResult model tests
# ---------------------------------------------------------------------------


class TestValidationResult:
    """Pydantic model validation and factory tests."""

    def test_default_construction(self) -> None:
        result = ValidationResult()
        assert result.passed is False
        assert result.retry_required is False
        assert result.severity == ValidationSeverity.NONE
        assert result.confidence_score == 0.0
        assert result.failed_checks == []
        assert result.passed_checks == []

    def test_full_construction(self) -> None:
        result = ValidationResult(
            passed=True,
            reason="All checks passed.",
            retry_required=False,
            severity=ValidationSeverity.NONE,
            confidence_score=0.92,
            failed_checks=[],
            passed_checks=[
                ValidationCheck(name="confidence_score", passed=True, detail="0.92 >= 0.60"),
                ValidationCheck(name="grounded", passed=True, detail="fully_grounded"),
                ValidationCheck(name="complete", passed=True, detail="answer is complete"),
                ValidationCheck(name="relevant", passed=True, detail="answer is relevant"),
            ],
        )
        assert result.passed is True
        assert result.retry_required is False
        assert len(result.passed_checks) == 4

    def test_error_result_factory(self) -> None:
        result = ValidationResult.error_result(error="Division by zero")
        assert result.passed is False
        assert result.retry_required is True
        assert result.severity == ValidationSeverity.MAJOR
        assert result.confidence_score == 0.0
        assert len(result.failed_checks) == 1
        assert result.failed_checks[0].name == "validator_internal"
        assert "Division by zero" in result.reason

    def test_to_log_dict(self) -> None:
        result = ValidationResult(
            passed=False,
            reason="Failed: grounded, complete",
            retry_required=True,
            severity=ValidationSeverity.MAJOR,
            confidence_score=0.45,
            failed_checks=[
                ValidationCheck(name="grounded", passed=False, detail="not grounded"),
                ValidationCheck(name="complete", passed=False, detail="incomplete"),
            ],
            passed_checks=[
                ValidationCheck(name="confidence_score", passed=True, detail="ok"),
                ValidationCheck(name="relevant", passed=True, detail="ok"),
            ],
        )
        log = result.to_log_dict()
        assert log == {
            "passed": False,
            "retry_required": True,
            "severity": "major",
            "confidence": 0.45,
            "failed_count": 2,
            "failed_names": ["grounded", "complete"],
        }

    def test_model_dump_for_graph(self) -> None:
        """model_dump() should produce a JSON-serialisable dict for AgentState."""
        result = ValidationResult(
            passed=True,
            reason="All good.",
            severity=ValidationSeverity.NONE,
            confidence_score=0.88,
            passed_checks=[
                ValidationCheck(name="confidence_score", passed=True, detail="ok"),
            ],
        )
        dumped = result.model_dump()
        assert isinstance(dumped, dict)
        assert dumped["passed"] is True
        assert dumped["severity"] == "none"
        # Round-trip through JSON
        import json
        assert json.loads(json.dumps(dumped)) == dumped


# ---------------------------------------------------------------------------
# AnswerValidator tests
# ---------------------------------------------------------------------------


class TestAnswerValidator:
    """Tests for AnswerValidator.validate() with various reflection inputs."""

    # -- happy path --------------------------------------------------------

    def test_valid_answer_passes_all_checks(self) -> None:
        """A high-quality, fully-grounded answer passes every check."""
        validator = AnswerValidator(thresholds=_default_thresholds())
        reflection = _make_reflection(
            answer_quality=AnswerQuality.EXCELLENT,
            grounded=GroundedStatus.FULLY_GROUNDED,
            complete=True,
            relevant=True,
            confidence_score=0.92,
        )

        result = validator.validate(reflection)

        assert result.passed is True
        assert result.retry_required is False
        assert result.severity == ValidationSeverity.NONE
        assert len(result.failed_checks) == 0
        assert len(result.passed_checks) == 4

    def test_lenient_thresholds_pass_everything(self) -> None:
        """Even a terrible answer passes when thresholds are lenient."""
        validator = AnswerValidator(thresholds=_lenient_thresholds())
        reflection = _make_reflection(
            answer_quality=AnswerQuality.INADEQUATE,
            grounded=GroundedStatus.UNGROUNDED,
            complete=False,
            relevant=False,
            confidence_score=0.1,
        )

        result = validator.validate(reflection)

        # Only confidence_score is checked (min 0.3 → fails at 0.1)
        # Other checks are disabled
        assert result.passed is False  # confidence 0.1 < 0.3
        assert len(result.failed_checks) == 1
        assert result.failed_checks[0].name == "confidence_score"

    # -- low confidence ----------------------------------------------------

    def test_low_confidence_fails(self) -> None:
        """Confidence below threshold should fail the check."""
        validator = AnswerValidator(thresholds=_default_thresholds())
        reflection = _make_reflection(
            confidence_score=0.45,  # below 0.6
            grounded=GroundedStatus.FULLY_GROUNDED,
            complete=True,
            relevant=True,
        )

        result = validator.validate(reflection)

        assert result.passed is False
        assert result.retry_required is True
        assert result.severity == ValidationSeverity.MINOR
        assert any(c.name == "confidence_score" for c in result.failed_checks)

    def test_confidence_exactly_at_threshold_passes(self) -> None:
        """Confidence == threshold should pass (>= check)."""
        validator = AnswerValidator(thresholds=_default_thresholds())  # min=0.6
        reflection = _make_reflection(confidence_score=0.6)

        result = validator.validate(reflection)
        confidence_check = next(
            c for c in result.passed_checks if c.name == "confidence_score"
        )
        assert confidence_check.passed is True

    # -- hallucinated / ungrounded -----------------------------------------

    def test_ungrounded_answer_fails(self) -> None:
        """An ungrounded answer triggers a MAJOR severity failure."""
        validator = AnswerValidator(thresholds=_default_thresholds())
        reflection = _make_reflection(
            answer_quality=AnswerQuality.INADEQUATE,
            grounded=GroundedStatus.UNGROUNDED,
            complete=False,
            relevant=True,
            confidence_score=0.8,
        )

        result = validator.validate(reflection)

        assert result.passed is False
        assert result.severity == ValidationSeverity.MAJOR
        assert any(c.name == "grounded" for c in result.failed_checks)
        assert any(c.name == "complete" for c in result.failed_checks)

    def test_partially_grounded_fails(self) -> None:
        """Partially grounded is not good enough when require_grounded=True."""
        validator = AnswerValidator(thresholds=_default_thresholds())
        reflection = _make_reflection(
            grounded=GroundedStatus.PARTIALLY_GROUNDED,
            confidence_score=0.8,
        )

        result = validator.validate(reflection)
        grounded_check = next(
            c for c in result.failed_checks if c.name == "grounded"
        )
        assert "must be fully_grounded" in grounded_check.detail

    def test_unknown_grounding_fails_when_required(self) -> None:
        """GroundedStatus.UNKNOWN should fail the grounded check."""
        validator = AnswerValidator(thresholds=_default_thresholds())
        reflection = _make_reflection(
            grounded=GroundedStatus.UNKNOWN,
            confidence_score=0.8,
        )

        result = validator.validate(reflection)
        assert any(c.name == "grounded" for c in result.failed_checks)

    # -- incomplete --------------------------------------------------------

    def test_incomplete_answer_fails(self) -> None:
        """An incomplete answer should fail the complete check."""
        validator = AnswerValidator(thresholds=_default_thresholds())
        reflection = _make_reflection(
            complete=False,
            confidence_score=0.8,
            grounded=GroundedStatus.FULLY_GROUNDED,
        )

        result = validator.validate(reflection)
        assert result.passed is False
        assert result.severity == ValidationSeverity.MAJOR
        assert any(c.name == "complete" for c in result.failed_checks)

    # -- irrelevant --------------------------------------------------------

    def test_irrelevant_answer_fails_critical(self) -> None:
        """An irrelevant answer is CRITICAL severity."""
        validator = AnswerValidator(thresholds=_default_thresholds())
        reflection = _make_reflection(
            answer_quality=AnswerQuality.IRRELEVANT,
            relevant=False,
            confidence_score=0.05,
        )

        result = validator.validate(reflection)

        assert result.passed is False
        assert result.severity == ValidationSeverity.CRITICAL
        assert any(c.name == "relevant" for c in result.failed_checks)

    # -- severity computation ----------------------------------------------

    def test_severity_none_when_all_pass(self) -> None:
        validator = AnswerValidator(thresholds=_default_thresholds())
        reflection = _make_reflection(confidence_score=0.9)
        result = validator.validate(reflection)
        assert result.severity == ValidationSeverity.NONE

    def test_severity_critical_when_irrelevant(self) -> None:
        validator = AnswerValidator(thresholds=_default_thresholds())
        reflection = _make_reflection(relevant=False)
        result = validator.validate(reflection)
        assert result.severity == ValidationSeverity.CRITICAL

    def test_severity_major_when_ungrounded(self) -> None:
        validator = AnswerValidator(thresholds=_default_thresholds())
        reflection = _make_reflection(
            grounded=GroundedStatus.UNGROUNDED,
            confidence_score=0.9,
        )
        result = validator.validate(reflection)
        assert result.severity == ValidationSeverity.MAJOR

    def test_severity_minor_when_only_confidence_low(self) -> None:
        validator = AnswerValidator(thresholds=_default_thresholds())
        reflection = _make_reflection(
            confidence_score=0.5,  # below 0.6
            grounded=GroundedStatus.FULLY_GROUNDED,
            complete=True,
            relevant=True,
        )
        result = validator.validate(reflection)
        assert result.severity == ValidationSeverity.MINOR

    # -- disabled checks ---------------------------------------------------

    def test_disabled_grounded_check_always_passes(self) -> None:
        """When require_grounded=False, ungrounded answers still pass that check."""
        thresholds = ValidationThresholds(
            min_confidence_score=0.6,
            require_grounded=False,  # disabled
            require_completeness=True,
            require_relevance=True,
        )
        validator = AnswerValidator(thresholds=thresholds)
        reflection = _make_reflection(
            grounded=GroundedStatus.UNGROUNDED,
            confidence_score=0.8,
        )

        result = validator.validate(reflection)
        assert not any(c.name == "grounded" for c in result.failed_checks)

    def test_disabled_completeness_check_always_passes(self) -> None:
        thresholds = ValidationThresholds(
            min_confidence_score=0.6,
            require_grounded=True,
            require_completeness=False,  # disabled
            require_relevance=True,
        )
        validator = AnswerValidator(thresholds=thresholds)
        reflection = _make_reflection(
            complete=False,
            confidence_score=0.8,
            grounded=GroundedStatus.FULLY_GROUNDED,
        )

        result = validator.validate(reflection)
        assert not any(c.name == "complete" for c in result.failed_checks)

    # -- validator failure -------------------------------------------------

    def test_validator_exception_returns_error_result(self) -> None:
        """If the validator itself throws, return a pessimistic result."""
        validator = AnswerValidator(thresholds=_default_thresholds())

        # Pass a broken reflection with invalid enum value
        result = validator.validate("not_a_reflection_result")  # type: ignore[arg-type]

        # Should catch the exception and return error_result
        assert result.passed is False
        assert result.retry_required is True
        assert result.severity == ValidationSeverity.MAJOR

    # -- threshold defaults ------------------------------------------------

    def test_default_thresholds_are_production_grade(self) -> None:
        """Default ValidationThresholds should require 0.6 confidence + all checks."""
        t = ValidationThresholds()
        assert t.min_confidence_score == 0.6
        assert t.require_grounded is True
        assert t.require_completeness is True
        assert t.require_relevance is True

    def test_thresholds_are_immutable(self) -> None:
        """ValidationThresholds is a frozen dataclass."""
        import dataclasses

        t = ValidationThresholds(min_confidence_score=0.7)
        with pytest.raises(dataclasses.FrozenInstanceError):
            t.min_confidence_score = 0.9  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Validation Severity enum tests
# ---------------------------------------------------------------------------


class TestValidationSeverity:
    def test_values(self) -> None:
        assert set(ValidationSeverity) == {
            ValidationSeverity.NONE,
            ValidationSeverity.MINOR,
            ValidationSeverity.MAJOR,
            ValidationSeverity.CRITICAL,
        }

    def test_str_enum_comparison(self) -> None:
        assert ValidationSeverity("minor") == ValidationSeverity.MINOR
        assert ValidationSeverity("major") == ValidationSeverity.MAJOR


# ---------------------------------------------------------------------------
# ValidationCheck model tests
# ---------------------------------------------------------------------------


class TestValidationCheck:
    def test_passing_check(self) -> None:
        check = ValidationCheck(
            name="confidence_score",
            passed=True,
            detail="0.85 >= 0.60",
        )
        assert check.passed is True
        assert "confidence_score" in check.name

    def test_failing_check(self) -> None:
        check = ValidationCheck(
            name="grounded",
            passed=False,
            detail="status=ungrounded — must be fully_grounded",
        )
        assert check.passed is False
        assert "ungrounded" in check.detail


# ---------------------------------------------------------------------------
# integration-style: validation node
# ---------------------------------------------------------------------------


class TestValidationNodeIntegration:
    """Lightweight tests that exercise the validation_node function directly."""

    def test_validation_node_stores_result(self) -> None:
        """Simulate the validation_node function with a mocked validator."""
        from app.agent.nodes import _services, validation_node

        validator = AnswerValidator(thresholds=_default_thresholds())
        _services["validator"] = validator

        reflection = _make_reflection(
            answer_quality=AnswerQuality.EXCELLENT,
            grounded=GroundedStatus.FULLY_GROUNDED,
            complete=True,
            relevant=True,
            confidence_score=0.92,
        )

        state: dict = {
            "question": "What is FAISS?",
            "reflection_result": reflection.model_dump(),
            "answer": "FAISS is a library for efficient similarity search.",
            "executed_nodes": ["planner", "retrieve", "generate", "reflection"],
        }

        result = validation_node(state)  # type: ignore[arg-type]

        assert "validation_result" in result
        vr = result["validation_result"]
        assert vr["passed"] is True
        assert vr["retry_required"] is False
        assert vr["severity"] == "none"
        assert "validation" in result["executed_nodes"]

    def test_validation_node_fails_on_hallucinated_answer(self) -> None:
        """The validation node should produce a failing result for ungrounded answers."""
        from app.agent.nodes import _services, validation_node

        validator = AnswerValidator(thresholds=_default_thresholds())
        _services["validator"] = validator

        reflection = _make_reflection(
            answer_quality=AnswerQuality.INADEQUATE,
            grounded=GroundedStatus.UNGROUNDED,
            complete=False,
            relevant=True,
            confidence_score=0.2,
        )

        state: dict = {
            "question": "What is FAISS?",
            "reflection_result": reflection.model_dump(),
            "executed_nodes": ["planner", "retrieve", "generate", "reflection"],
        }

        result = validation_node(state)  # type: ignore[arg-type]

        vr = result["validation_result"]
        assert vr["passed"] is False
        assert vr["retry_required"] is True

    def test_validation_node_with_missing_reflection(self) -> None:
        """Gracefully handles an empty reflection_result dict."""
        from app.agent.nodes import _services, validation_node

        validator = AnswerValidator(thresholds=_default_thresholds())
        _services["validator"] = validator

        state: dict = {
            "question": "What is FAISS?",
            "reflection_result": {},  # empty — default ReflectionResult
            "executed_nodes": ["generate"],
        }

        result = validation_node(state)  # type: ignore[arg-type]

        assert "validation_result" in result
        # Default ReflectionResult has confidence=0.5, complete=False
        # With default thresholds (min 0.6): confidence fails
        vr = result["validation_result"]
        assert vr["passed"] is False

    def test_validation_node_engine_not_configured(self) -> None:
        """Raises RuntimeError if validator was never registered."""
        from app.agent.nodes import _services, validation_node

        _services.pop("validator", None)

        state: dict = {
            "question": "What is FAISS?",
            "reflection_result": _make_reflection().model_dump(),
            "executed_nodes": ["generate", "reflection"],
        }

        import pytest

        with pytest.raises(RuntimeError, match="validator"):
            validation_node(state)  # type: ignore[arg-type]
