"""In-memory job model for background indexing."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum


class JobStatus(StrEnum):
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


@dataclass
class Job:
    """Tracks the lifecycle of a single background indexing run."""

    job_id: str
    status: JobStatus = JobStatus.QUEUED
    created_at: str = field(default_factory=lambda: _utcnow())
    started_at: str | None = None
    completed_at: str | None = None
    files_processed: int = 0
    chunks_created: int = 0
    embeddings_generated: int = 0
    error_message: str | None = None


def new_job_id() -> str:
    return uuid.uuid4().hex[:12]


def _utcnow() -> str:
    return datetime.now(UTC).isoformat()
