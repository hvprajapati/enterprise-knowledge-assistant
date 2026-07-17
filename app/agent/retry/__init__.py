"""Agent Retry — intelligent pipeline recovery after validation.

The retry sub-package decides whether (and how) to loop back through
the pipeline when the generated answer fails quality checks.  It is
a **targeted recovery** layer — it never blindly re-runs everything.

Components
----------
- ``RetryDecision`` — structured Pydantic model for the retry decision.
- ``RetryStrategy`` — enum of pipeline re-entry points.
- ``RetryEngine`` — deterministic decision engine.
- ``retry_node`` — LangGraph node (in ``app.agent.nodes``).
"""

from __future__ import annotations

from app.agent.retry.models import RetryDecision, RetryStrategy
from app.agent.retry.retry_engine import RetryEngine

__all__ = [
    "RetryDecision",
    "RetryEngine",
    "RetryStrategy",
]
