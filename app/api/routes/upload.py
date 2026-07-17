"""File upload endpoints.

POST   /api/v1/upload         — accept a document, trigger background indexing
GET    /api/v1/uploads        — list uploaded files
DELETE /api/v1/uploads/{name} — remove an uploaded file
"""

from __future__ import annotations

import logging
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, UploadFile

from app.api.dependencies import build_index_builder, get_job_manager
from app.api.schemas import UploadedFileInfo, UploadResponse
from app.config.settings import settings
from app.jobs.manager import JobManager

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Upload"])


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _allowed_extension(filename: str) -> bool:
    exts = {
        e.strip().lower()
        for e in settings.supported_upload_extensions.split(",")
    }
    return Path(filename).suffix.lower() in exts


def _ensure_upload_dir() -> Path:
    directory = Path(settings.upload_directory)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


# ---------------------------------------------------------------------------
# POST /upload
# ---------------------------------------------------------------------------


@router.post(
    "/upload",
    response_model=UploadResponse,
    summary="Upload a document and trigger background indexing",
    description=(
        "Accepts a single file via ``multipart/form-data``.  Validates "
        "the extension, size, and emptiness, then saves it to the "
        "configured upload directory and schedules a background "
        "indexing job."
    ),
)
async def upload_file(
    background: BackgroundTasks,
    jobs: Annotated[JobManager, Depends(get_job_manager)],
    file: Annotated[UploadFile, File(description="Document to upload")],
) -> UploadResponse:
    # -- validate ---------------------------------------------------------
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename is required.")

    if not _allowed_extension(file.filename):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {Path(file.filename).suffix}. "
            f"Allowed: {settings.supported_upload_extensions}",
        )

    content = await file.read()

    if not content:
        raise HTTPException(status_code=400, detail="File is empty.")

    if len(content) > settings.max_upload_size:
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds maximum size of {settings.max_upload_size} bytes.",
        )

    # -- save --------------------------------------------------------------
    upload_dir = _ensure_upload_dir()
    dest = upload_dir / file.filename

    if dest.exists():
        raise HTTPException(
            status_code=409,
            detail=f"File '{file.filename}' already exists. "
            f"Delete it first via DELETE /api/v1/uploads/{file.filename}.",
        )

    dest.write_bytes(content)
    logger.info("Upload saved — filename=%s  size=%d", file.filename, len(content))

    # -- schedule background indexing --------------------------------------
    job = jobs.create_job()
    index_path = Path(settings.index_storage_path)

    background.add_task(
        _run_indexing_for_upload,
        job_id=job.job_id,
        directory=str(upload_dir),
        index_path=str(index_path),
    )

    logger.info(
        "Upload indexing job scheduled — job_id=%s  directory=%s",
        job.job_id,
        upload_dir,
    )

    return UploadResponse(
        filename=file.filename,
        job_id=job.job_id,
        status=job.status.value,
    )


# ---------------------------------------------------------------------------
# GET /uploads
# ---------------------------------------------------------------------------


@router.get(
    "/uploads",
    response_model=list[UploadedFileInfo],
    summary="List all uploaded files",
)
async def list_uploads() -> list[UploadedFileInfo]:
    upload_dir = _ensure_upload_dir()
    files: list[UploadedFileInfo] = []

    for entry in sorted(upload_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        if entry.is_file():
            stat = entry.stat()
            files.append(
                UploadedFileInfo(
                    filename=entry.name,
                    size=stat.st_size,
                    uploaded_at=datetime.fromtimestamp(stat.st_mtime, tz=UTC).isoformat(),
                )
            )

    return files


# ---------------------------------------------------------------------------
# DELETE /uploads/{filename}
# ---------------------------------------------------------------------------


@router.delete(
    "/uploads/{filename:path}",
    summary="Delete an uploaded file",
    description=(
        "Removes the file from the upload directory.  "
        "**Note:** vectors and metadata already indexed from this file "
        "are NOT automatically removed from FAISS or SQLite.  "
        "Re-run ``POST /api/v1/index`` on the upload directory to "
        "rebuild the index without the deleted file."
    ),
)
async def delete_upload(filename: str) -> dict[str, str]:
    upload_dir = _ensure_upload_dir()
    target = (upload_dir / filename).resolve()

    # Prevent path traversal
    if not str(target).startswith(str(upload_dir.resolve())):
        raise HTTPException(status_code=400, detail="Invalid filename.")

    if not target.is_file():
        raise HTTPException(status_code=404, detail=f"File not found: {filename}")

    os.remove(target)
    logger.info("Upload deleted — filename=%s", filename)

    return {"detail": f"Deleted: {filename}"}


# ---------------------------------------------------------------------------
# background worker
# ---------------------------------------------------------------------------


def _run_indexing_for_upload(
    job_id: str,
    directory: str,
    index_path: str,
) -> None:
    """Rebuild the FAISS index from *directory* and update job status."""
    from app.jobs.manager import job_manager as jm

    jm.transition_to_running(job_id)
    logger.info("Upload indexing started — job_id=%s  directory=%s", job_id, directory)

    try:
        builder = build_index_builder()
        stats = builder.build(
            input_folder=Path(directory),
            index_path=Path(index_path),
        )
        jm.transition_to_completed(
            job_id,
            files_processed=stats.files_processed,
            chunks_created=stats.chunks_created,
            embeddings_generated=stats.embeddings_generated,
        )
    except Exception as exc:
        logger.exception("Upload indexing failed — job_id=%s", job_id)
        jm.transition_to_failed(job_id, str(exc))
