"""Unit tests for the Retry Engine and Retry Node.

Covers:
- NO_RETRY (validation passed)
- RETRIEVE_AGAIN (major severity)
- GENERATE_AGAIN (minor severity)
- FULL_PIPELINE (critical severity)
- retry limit reached
- retry engine failure / fallback
- RetryDecision model serialization
- retry node integration
"""

from __future__ import annotations

import pytest

from app.agent.reflection.models import (
    AnswerQuality,
    GroundedStatus,
    ReflectionResult,
)
from app.agent.retry import RetryDecision, RetryEngine, RetryStrategy
from app.agent.retry.strategies import (
    STRATEGY_NODE_MAP,
    resolve_retry_strategy,
)
from app.agent.validation.models import (
    ValidationCheck,
    ValidationResult,
    ValidationSeverity,
)

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _make_validation(
    *,
    passed: bool = True,
    severity: ValidationSeverity = ValidationSeverity.NONE,
    retry_required: bool = False,
    confidence_score: float = 0.85,
    failed_checks: list[ValidationCheck] | None = None,
    passed_checks: list[ValidationCheck] | None = None,
) -> ValidationResult:
    """Build a ValidationResult with sensible defaults."""
    return ValidationResult(
        passed=passed,
        reason="Test validation.",
        retry_required=retry_required,
        severity=severity,
        confidence_score=confidence_score,
        failed_checks=failed_checks or [],
        passed_checks=passed_checks or [],
    )


def _make_reflection(
    *,
    answer_quality: AnswerQuality = AnswerQuality.GOOD,
    grounded: GroundedStatus = GroundedStatus.FULLY_GROUNDED,
    complete: bool = True,
    relevant: bool = True,
    confidence_score: float = 0.85,
    recommendations: list[str] | None = None,
) -> ReflectionResult:
    """Build a ReflectionResult with sensible defaults."""
    return ReflectionResult(
        answer_quality=answer_quality,
        grounded=grounded,
        complete=complete,
        relevant=relevant,
        confidence_score=confidence_score,
        missing_information=[],
        recommendations=recommendations or [],
        reasoning="Test reflection.",
    )


# ---------------------------------------------------------------------------
# RetryDecision model tests
# ---------------------------------------------------------------------------


class TestRetryDecision:
    """Pydantic model validation and factory tests."""

    def test_default_construction(self) -> None:
        decision = RetryDecision()
        assert decision.should_retry is False
        assert decision.strategy == RetryStrategy.NO_RETRY
        assert decision.next_node == "__end__"
        assert decision.retry_count == 0
        assert decision.max_retries == 2

    def test_no_retry_factory(self) -> None:
        decision = RetryDecision.no_retry(
            reason="Looks good.",
            retry_count=0,
            max_retries=2,
        )
        assert decision.should_retry is False
        assert decision.strategy == RetryStrategy.NO_RETRY
        assert decision.next_node == "__end__"
        assert "Looks good" in decision.reason

    def test_retry_limit_reached_factory(self) -> None:
        decision = RetryDecision.retry_limit_reached(
            retry_count=2,
            max_retries=2,
        )
        assert decision.should_retry is False
        assert decision.strategy == RetryStrategy.NO_RETRY
        assert "Retry limit reached" in decision.reason
        assert "2/2" in decision.reason

    def test_error_fallback_factory(self) -> None:
        decision = RetryDecision.error_fallback(retry_count=1)
        assert decision.should_retry is False
        assert decision.strategy == RetryStrategy.NO_RETRY
        assert "error" in decision.reason.lower()

    def test_full_retry_decision(self) -> None:
        decision = RetryDecision(
            should_retry=True,
            strategy=RetryStrategy.GENERATE_AGAIN,
            reason="Low confidence — regenerate.",
            retry_count=0,
            max_retries=2,
            next_node="generate_node",
        )
        assert decision.should_retry is True
        assert decision.next_node == "generate_node"

    def test_to_log_dict(self) -> None:
        decision = RetryDecision(
            should_retry=True,
            strategy=RetryStrategy.RETRIEVE_AGAIN,
            reason="Re-retrieve with broader search.",
            retry_count=1,
            max_retries=2,
            next_node="retrieve_node",
        )
        log = decision.to_log_dict()
        assert log == {
            "should_retry": True,
            "strategy": "retrieve_again",
            "retry_count": 1,
            "max_retries": 2,
            "next_node": "retrieve_node",
            "reason": "Re-retrieve with broader search.",
        }

    def test_model_dump_for_graph(self) -> None:
        decision = RetryDecision(
            should_retry=False,
            strategy=RetryStrategy.NO_RETRY,
            reason="All good.",
            retry_count=0,
            max_retries=2,
            next_node="__end__",
        )
        dumped = decision.model_dump()
        assert isinstance(dumped, dict)
        assert dumped["strategy"] == "no_retry"
        assert dumped["next_node"] == "__end__"


# ---------------------------------------------------------------------------
# RetryStrategy enum tests
# ---------------------------------------------------------------------------


class TestRetryStrategy:
    def test_values(self) -> None:
        assert set(RetryStrategy) == {
            RetryStrategy.NO_RETRY,
            RetryStrategy.RETRIEVE_AGAIN,
            RetryStrategy.GENERATE_AGAIN,
            RetryStrategy.FULL_PIPELINE,
        }

    def test_str_node_map(self) -> None:
        """Every strategy must have a valid node mapping."""
        assert STRATEGY_NODE_MAP[RetryStrategy.NO_RETRY] == "__end__"
        assert STRATEGY_NODE_MAP[RetryStrategy.RETRIEVE_AGAIN] == "retrieve_node"
        assert STRATEGY_NODE_MAP[RetryStrategy.GENERATE_AGAIN] == "generate_node"
        assert STRATEGY_NODE_MAP[RetryStrategy.FULL_PIPELINE] == "planner_node"


# ---------------------------------------------------------------------------
# resolve_retry_strategy tests
# ---------------------------------------------------------------------------


class TestResolveRetryStrategy:
    """Tests for the core strategy mapping function."""

    def test_none_severity_returns_no_retry(self) -> None:
        decision = resolve_retry_strategy(
            ValidationSeverity.NONE,
            retry_count=0,
            max_retries=2,
        )
        assert decision.should_retry is False
        assert decision.strategy == RetryStrategy.NO_RETRY
        assert decision.next_node == "__end__"

    def test_minor_severity_returns_generate_again(self) -> None:
        decision = resolve_retry_strategy(
            ValidationSeverity.MINOR,
            retry_count=0,
            max_retries=2,
        )
        assert decision.should_retry is True
        assert decision.strategy == RetryStrategy.GENERATE_AGAIN
        assert decision.next_node == "generate_node"

    def test_major_severity_returns_retrieve_again(self) -> None:
        decision = resolve_retry_strategy(
            ValidationSeverity.MAJOR,
            retry_count=0,
            max_retries=2,
        )
        assert decision.should_retry is True
        assert decision.strategy == RetryStrategy.RETRIEVE_AGAIN
        assert decision.next_node == "retrieve_node"

    def test_critical_severity_returns_full_pipeline(self) -> None:
        decision = resolve_retry_strategy(
            ValidationSeverity.CRITICAL,
            retry_count=0,
            max_retries=2,
        )
        assert decision.should_retry is True
        assert decision.strategy == RetryStrategy.FULL_PIPELINE
        assert decision.next_node == "planner_node"

    def test_retry_limit_reached_forces_no_retry(self) -> None:
        """At retry_count=2, max=2, should force NO_RETRY regardless of severity."""
        decision = resolve_retry_strategy(
            ValidationSeverity.CRITICAL,  # would normally retry
            retry_count=2,
            max_retries=2,
        )
        assert decision.should_retry is False
        assert decision.strategy == RetryStrategy.NO_RETRY
        assert "Retry limit reached" in decision.reason

    def test_retry_count_below_limit_allows_retry(self) -> None:
        """At retry_count=1, max=2, retry is still allowed."""
        decision = resolve_retry_strategy(
            ValidationSeverity.MAJOR,
            retry_count=1,
            max_retries=2,
        )
        assert decision.should_retry is True

    def test_retry_count_above_limit_forces_no_retry(self) -> None:
        """At retry_count=3, max=2, should force NO_RETRY."""
        decision = resolve_retry_strategy(
            ValidationSeverity.MAJOR,
            retry_count=3,
            max_retries=2,
        )
        assert decision.should_retry is False
        assert "Retry limit reached" in decision.reason


# ---------------------------------------------------------------------------
# RetryEngine tests
# ---------------------------------------------------------------------------


class TestRetryEngine:
    """Tests for RetryEngine.decide_retry()."""

    def test_validation_passed_returns_no_retry(self) -> None:
        engine = RetryEngine(max_retries=2)
        validation = _make_validation(
            passed=True,
            severity=ValidationSeverity.NONE,
        )
        reflection = _make_reflection()

        decision = engine.decide_retry(
            validation=validation,
            reflection=reflection,
            retry_count=0,
        )

        assert decision.should_retry is False
        assert decision.strategy == RetryStrategy.NO_RETRY
        assert decision.next_node == "__end__"

    def test_low_confidence_retrieves_again(self) -> None:
        """MAJOR severity → RETRIEVE_AGAIN."""
        engine = RetryEngine(max_retries=2)
        validation = _make_validation(
            passed=False,
            severity=ValidationSeverity.MAJOR,
            retry_required=True,
            confidence_score=0.3,
            failed_checks=[
                ValidationCheck(name="grounded", passed=False, detail="not grounded"),
            ],
        )
        reflection = _make_reflection(
            grounded=GroundedStatus.UNGROUNDED,
            confidence_score=0.3,
        )

        decision = engine.decide_retry(
            validation=validation,
            reflection=reflection,
            retry_count=0,
        )

        assert decision.should_retry is True
        assert decision.strategy == RetryStrategy.RETRIEVE_AGAIN
        assert decision.next_node == "retrieve_node"

    def test_minor_issue_generates_again(self) -> None:
        """MINOR severity → GENERATE_AGAIN."""
        engine = RetryEngine(max_retries=2)
        validation = _make_validation(
            passed=False,
            severity=ValidationSeverity.MINOR,
            retry_required=True,
            confidence_score=0.55,
            failed_checks=[
                ValidationCheck(
                    name="confidence_score",
                    passed=False,
                    detail="0.55 < 0.60",
                ),
            ],
        )
        reflection = _make_reflection(confidence_score=0.55)

        decision = engine.decide_retry(
            validation=validation,
            reflection=reflection,
            retry_count=0,
        )

        assert decision.should_retry is True
        assert decision.strategy == RetryStrategy.GENERATE_AGAIN
        assert decision.next_node == "generate_node"

    def test_critical_issue_full_pipeline(self) -> None:
        """CRITICAL severity → FULL_PIPELINE."""
        engine = RetryEngine(max_retries=2)
        validation = _make_validation(
            passed=False,
            severity=ValidationSeverity.CRITICAL,
            retry_required=True,
            confidence_score=0.05,
            failed_checks=[
                ValidationCheck(name="relevant", passed=False, detail="irrelevant"),
            ],
        )
        reflection = _make_reflection(
            answer_quality=AnswerQuality.IRRELEVANT,
            relevant=False,
            confidence_score=0.05,
        )

        decision = engine.decide_retry(
            validation=validation,
            reflection=reflection,
            retry_count=0,
        )

        assert decision.should_retry is True
        assert decision.strategy == RetryStrategy.FULL_PIPELINE
        assert decision.next_node == "planner_node"

    def test_retry_limit_reached_after_max_retries(self) -> None:
        """After max_retries, even CRITICAL severity should stop retrying."""
        engine = RetryEngine(max_retries=2)
        validation = _make_validation(
            passed=False,
            severity=ValidationSeverity.CRITICAL,
            retry_required=True,
        )
        reflection = _make_reflection()

        decision = engine.decide_retry(
            validation=validation,
            reflection=reflection,
            retry_count=2,
        )

        assert decision.should_retry is False
        assert decision.strategy == RetryStrategy.NO_RETRY
        assert "2/2" in decision.reason

    def test_reflection_recommendations_are_logged_on_retry(self) -> None:
        """Recommendations should be available (logged) when retrying (no crash)."""
        engine = RetryEngine(max_retries=2)
        validation = _make_validation(
            passed=False,
            severity=ValidationSeverity.MAJOR,
            retry_required=True,
        )
        reflection = _make_reflection(
            grounded=GroundedStatus.UNGROUNDED,
            recommendations=[
                "Use broader search terms.",
                "Include more context.",
            ],
        )

        decision = engine.decide_retry(
            validation=validation,
            reflection=reflection,
            retry_count=0,
        )

        assert decision.should_retry is True
        # recommendations on reflection are intact (logged internally)

    def test_engine_rejects_negative_max_retries(self) -> None:
        with pytest.raises(ValueError, match="max_retries"):
            RetryEngine(max_retries=-1)

    def test_engine_accepts_zero_max_retries(self) -> None:
        """max_retries=0 means no retries ever."""
        engine = RetryEngine(max_retries=0)
        validation = _make_validation(
            passed=False,
            severity=ValidationSeverity.CRITICAL,
            retry_required=True,
        )
        reflection = _make_reflection()

        decision = engine.decide_retry(
            validation=validation,
            reflection=reflection,
            retry_count=0,
        )

        # retry_count(0) >= max_retries(0) → NO_RETRY
        assert decision.should_retry is False

    # -- retry count increment scenarios -----------------------------------

    def test_retry_count_0_at_first_retry(self) -> None:
        """First retry: count=0, allowed."""
        engine = RetryEngine(max_retries=2)
        decision = engine.decide_retry(
            validation=_make_validation(
                passed=False,
                severity=ValidationSeverity.MAJOR,
                retry_required=True,
            ),
            reflection=_make_reflection(),
            retry_count=0,
        )
        assert decision.should_retry is True
        assert decision.retry_count == 0

    def test_retry_count_1_second_retry_allowed(self) -> None:
        """Second retry: count=1, max=2, allowed."""
        engine = RetryEngine(max_retries=2)
        decision = engine.decide_retry(
            validation=_make_validation(
                passed=False,
                severity=ValidationSeverity.MAJOR,
                retry_required=True,
            ),
            reflection=_make_reflection(),
            retry_count=1,
        )
        assert decision.should_retry is True


# ---------------------------------------------------------------------------
# integration-style: retry node
# ---------------------------------------------------------------------------


class TestRetryNodeIntegration:
    """Lightweight tests that exercise the retry_node function directly."""

    def test_retry_node_no_retry_when_passed(self) -> None:
        """When validation passes, retry_node should return NO_RETRY."""
        from app.agent.nodes import _services, retry_node

        engine = RetryEngine(max_retries=2)
        _services["retry_engine"] = engine

        validation = _make_validation(
            passed=True,
            severity=ValidationSeverity.NONE,
        )
        reflection = _make_reflection()

        state: dict = {
            "question": "What is FAISS?",
            "reflection_result": reflection.model_dump(),
            "validation_result": validation.model_dump(),
            "retry_count": 0,
            "answer": "FAISS is a library for similarity search.",
            "executed_nodes": [
                "planner",
                "retrieve",
                "generate",
                "reflection",
                "validation",
            ],
        }

        result = retry_node(state)  # type: ignore[arg-type]

        assert "retry_decision" in result
        rd = result["retry_decision"]
        assert rd["should_retry"] is False
        assert rd["strategy"] == "no_retry"
        assert rd["next_node"] == "__end__"
        assert result["retry_count"] == 0  # not incremented
        assert "retry" in result["executed_nodes"]

    def test_retry_node_triggers_retrieve_again(self) -> None:
        """MAJOR severity should trigger RETRIEVE_AGAIN."""
        from app.agent.nodes import _services, retry_node

        engine = RetryEngine(max_retries=2)
        _services["retry_engine"] = engine

        validation = _make_validation(
            passed=False,
            severity=ValidationSeverity.MAJOR,
            retry_required=True,
            confidence_score=0.3,
        )
        reflection = _make_reflection(
            grounded=GroundedStatus.UNGROUNDED,
            confidence_score=0.3,
        )

        state: dict = {
            "question": "What is FAISS?",
            "reflection_result": reflection.model_dump(),
            "validation_result": validation.model_dump(),
            "retry_count": 0,
            "answer": "FAISS is a database.",
            "executed_nodes": [
                "planner",
                "retrieve",
                "generate",
                "reflection",
                "validation",
            ],
        }

        result = retry_node(state)  # type: ignore[arg-type]

        rd = result["retry_decision"]
        assert rd["should_retry"] is True
        assert rd["strategy"] == "retrieve_again"
        assert rd["next_node"] == "retrieve_node"
        assert result["retry_count"] == 1  # incremented

    def test_retry_node_increments_count_on_retry(self) -> None:
        """Retry count should increment when retrying."""
        from app.agent.nodes import _services, retry_node

        engine = RetryEngine(max_retries=2)
        _services["retry_engine"] = engine

        validation = _make_validation(
            passed=False,
            severity=ValidationSeverity.MINOR,
            retry_required=True,
        )
        reflection = _make_reflection()

        state: dict = {
            "question": "Q",
            "reflection_result": reflection.model_dump(),
            "validation_result": validation.model_dump(),
            "retry_count": 1,  # already retried once
            "executed_nodes": ["planner", "retrieve", "generate", "reflection", "validation"],
        }

        result = retry_node(state)  # type: ignore[arg-type]

        rd = result["retry_decision"]
        assert rd["should_retry"] is True
        assert result["retry_count"] == 2  # 1 → 2

    def test_retry_node_hits_limit_returns_no_retry(self) -> None:
        """When retry_count already at max, should return NO_RETRY."""
        from app.agent.nodes import _services, retry_node

        engine = RetryEngine(max_retries=2)
        _services["retry_engine"] = engine

        validation = _make_validation(
            passed=False,
            severity=ValidationSeverity.CRITICAL,
            retry_required=True,
        )
        reflection = _make_reflection()

        state: dict = {
            "question": "Q",
            "reflection_result": reflection.model_dump(),
            "validation_result": validation.model_dump(),
            "retry_count": 2,  # already at max
            "executed_nodes": [
                "planner",
                "retrieve",
                "generate",
                "reflection",
                "validation",
            ],
        }

        result = retry_node(state)  # type: ignore[arg-type]

        rd = result["retry_decision"]
        assert rd["should_retry"] is False
        assert rd["strategy"] == "no_retry"
        assert result["retry_count"] == 2  # not incremented beyond max

    def test_retry_node_engine_not_configured(self) -> None:
        """Raises RuntimeError if engine was never registered."""
        from app.agent.nodes import _services, retry_node

        _services.pop("retry_engine", None)

        state: dict = {
            "question": "Q",
            "reflection_result": _make_reflection().model_dump(),
            "validation_result": _make_validation().model_dump(),
            "retry_count": 0,
            "executed_nodes": ["reflection", "validation"],
        }

        with pytest.raises(RuntimeError, match="retry_engine"):
            retry_node(state)  # type: ignore[arg-type]
