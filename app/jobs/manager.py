"""In-memory job lifecycle manager — thread-safe.

Manages the lifecycle of background indexing jobs.  Jobs are stored
in a process-local dictionary protected by a threading lock so that
concurrent API requests and background workers can read/write safely.
"""

from __future__ import annotations

import logging
import threading
from datetime import UTC, datetime

from app.jobs.models import Job, JobStatus, new_job_id

logger = logging.getLogger(__name__)


class JobManager:
    """Create, query, and update background indexing jobs."""

    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # create
    # ------------------------------------------------------------------

    def create_job(self) -> Job:
        """Register a new job in QUEUED state and return it."""
        job = Job(job_id=new_job_id())
        with self._lock:
            self._jobs[job.job_id] = job
        logger.info("Job created — id=%s", job.job_id)
        return job

    # ------------------------------------------------------------------
    # read
    # ------------------------------------------------------------------

    def get_job(self, job_id: str) -> Job | None:
        """Return the job, or ``None`` if *job_id* is unknown."""
        with self._lock:
            return self._jobs.get(job_id)

    # ------------------------------------------------------------------
    # update
    # ------------------------------------------------------------------

    def transition_to_running(self, job_id: str) -> None:
        self._update(job_id, status=JobStatus.RUNNING, started_at=_utcnow())

    def transition_to_completed(
        self,
        job_id: str,
        files_processed: int,
        chunks_created: int,
        embeddings_generated: int,
    ) -> None:
        self._update(
            job_id,
            status=JobStatus.COMPLETED,
            completed_at=_utcnow(),
            files_processed=files_processed,
            chunks_created=chunks_created,
            embeddings_generated=embeddings_generated,
        )
        logger.info(
            "Job completed — id=%s  files=%d  chunks=%d  embeddings=%d",
            job_id,
            files_processed,
            chunks_created,
            embeddings_generated,
        )

    def transition_to_failed(self, job_id: str, error_message: str) -> None:
        self._update(
            job_id,
            status=JobStatus.FAILED,
            completed_at=_utcnow(),
            error_message=error_message,
        )
        logger.error("Job failed — id=%s  error=%s", job_id, error_message)

    # ------------------------------------------------------------------
    # internal
    # ------------------------------------------------------------------

    def _update(self, job_id: str, **kwargs: object) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                logger.warning("Update for unknown job: %s", job_id)
                return
            for key, value in kwargs.items():
                setattr(job, key, value)


def _utcnow() -> str:
    return datetime.now(UTC).isoformat()


# ------------------------------------------------------------------
# module-level singleton — created once at import time
# ------------------------------------------------------------------

job_manager = JobManager()
