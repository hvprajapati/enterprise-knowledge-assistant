"""POST /api/v1/index — background document indexing endpoint.

Accepts a directory path, returns a job ID immediately, and schedules
the actual indexing work via FastAPI ``BackgroundTasks``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, BackgroundTasks, Depends

from app.api.dependencies import build_index_builder, get_job_manager
from app.api.schemas import IndexJobResponse, IndexRequest, IndexResponse
from app.config.settings import settings
from app.indexing.index_builder import IndexBuilder
from app.jobs.manager import JobManager

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Index"])


# ---------------------------------------------------------------------------
# POST /index  (async — returns job ID immediately)
# ---------------------------------------------------------------------------


@router.post(
    "/index",
    response_model=IndexJobResponse,
    summary="Start a background indexing job",
    description=(
        "Validates *directory*, creates a job, and schedules indexing "
        "to run in the background.  Returns a job ID immediately — "
        "use ``GET /api/v1/jobs/{job_id}`` to track progress."
    ),
)
async def start_index_job(
    body: IndexRequest,
    background: BackgroundTasks,
    jobs: Annotated[JobManager, Depends(get_job_manager)],
) -> IndexJobResponse:
    directory = Path(body.directory)
    if not directory.is_dir():
        raise ValueError(f"Not a directory: {directory}")

    index_path = Path(settings.index_storage_path)

    # Create the job
    job = jobs.create_job()

    # Schedule the real work
    background.add_task(
        _run_indexing,
        job_id=job.job_id,
        input_folder=str(directory),
        index_path=str(index_path),
    )

    logger.info(
        "Background indexing job scheduled — id=%s  directory=%s  index_path=%s",
        job.job_id,
        directory,
        index_path,
    )

    return IndexJobResponse(job_id=job.job_id, status=job.status.value)


# ---------------------------------------------------------------------------
# POST /index/sync  (legacy behaviour for direct calls)
# ---------------------------------------------------------------------------


@router.post(
    "/index/sync",
    response_model=IndexResponse,
    summary="Run indexing synchronously (blocks until done)",
    description=(
        "Legacy synchronous behaviour.  Blocks the request until "
        "indexing is complete.  Prefer ``POST /index`` for large "
        "directories."
    ),
)
async def build_index_sync(
    body: IndexRequest,
    builder: Annotated[IndexBuilder, Depends(build_index_builder)],
) -> IndexResponse:
    directory = Path(body.directory)
    if not directory.is_dir():
        raise ValueError(f"Not a directory: {directory}")

    index_path = Path(settings.index_storage_path)

    logger.info("Sync index request — directory=%s  index_path=%s", directory, index_path)

    stats = builder.build(input_folder=directory, index_path=index_path)

    return IndexResponse(
        files_processed=stats.files_processed,
        chunks_created=stats.chunks_created,
        embeddings_generated=stats.embeddings_generated,
    )


# ---------------------------------------------------------------------------
# background worker (runs outside FastAPI DI scope)
# ---------------------------------------------------------------------------


def _run_indexing(
    job_id: str,
    input_folder: str,
    index_path: str,
) -> None:
    """Execute the indexing pipeline and update job status."""
    from app.jobs.manager import job_manager as jm

    jm.transition_to_running(job_id)
    logger.info("Background indexing started — id=%s  directory=%s", job_id, input_folder)

    try:
        builder = build_index_builder()
        stats = builder.build(
            input_folder=Path(input_folder),
            index_path=Path(index_path),
        )
        jm.transition_to_completed(
            job_id,
            files_processed=stats.files_processed,
            chunks_created=stats.chunks_created,
            embeddings_generated=stats.embeddings_generated,
        )

    except Exception as exc:
        logger.exception("Background indexing failed — id=%s", job_id)
        jm.transition_to_failed(job_id, str(exc))
