"""GET /api/v1/jobs/{job_id} — job status endpoint."""

from __future__ import annotations

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException

from app.api.dependencies import get_job_manager
from app.api.schemas import JobStatusResponse
from app.jobs.manager import JobManager

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Jobs"])


@router.get(
    "/jobs/{job_id}",
    response_model=JobStatusResponse,
    summary="Get the status of a background indexing job",
    description=(
        "Returns the current status, timestamps, and (when COMPLETED) "
        "the indexing statistics for the given *job_id*."
    ),
)
async def get_job_status(
    job_id: str,
    jobs: Annotated[JobManager, Depends(get_job_manager)],
) -> JobStatusResponse:
    job = jobs.get_job(job_id)

    if job is None:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")

    return JobStatusResponse(
        job_id=job.job_id,
        status=job.status.value,
        created_at=job.created_at,
        started_at=job.started_at,
        completed_at=job.completed_at,
        files_processed=job.files_processed,
        chunks_created=job.chunks_created,
        embeddings_generated=job.embeddings_generated,
        error_message=job.error_message,
    )
