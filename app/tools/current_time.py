"""Current time tool — returns the current date/time.

This is useful when the LLM needs to answer time-sensitive questions
(e.g. "What day is it?", "How many days until...?").
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from app.tools.base import BaseTool


class CurrentTimeTool(BaseTool):
    """Return the current UTC date and time."""

    @property
    def name(self) -> str:
        return "current-time"

    @property
    def description(self) -> str:
        return (
            "Return the current date and time in UTC. "
            "No arguments required. "
            "Use when the user asks about the current time, date, "
            "day of week, or needs a timestamp."
        )

    @property
    def schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {},
            "required": [],
        }

    def invoke(self, **kwargs: Any) -> dict[str, Any]:
        now = datetime.now(UTC)
        return {
            "result": now.isoformat(),
            "timestamp_unix": now.timestamp(),
            "day_of_week": now.strftime("%A"),
            "date": now.strftime("%Y-%m-%d"),
            "time": now.strftime("%H:%M:%S"),
            "timezone": "UTC",
        }
