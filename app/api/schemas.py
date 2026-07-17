"""Pydantic models for the REST API layer.

Internal models are re-exported where the API contract matches exactly.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

# Re-export — used by route modules that import from this package.
from app.query.models import QueryResponse  # noqa: F401

__all__ = [
    "ErrorResponse",
    "IndexJobResponse",
    "IndexRequest",
    "IndexResponse",
    "JobStatusResponse",
    "QueryRequest",
    "QueryResponse",
    "UploadResponse",
    "UploadedFileInfo",
]


# ---------------------------------------------------------------------------
# Query
# ---------------------------------------------------------------------------


class QueryRequest(BaseModel):
    """Incoming query request body."""

    question: str = Field(
        ...,
        min_length=1,
        description="The user's natural-language question.",
        examples=["What is Amazon Bedrock?"],
    )


# ---------------------------------------------------------------------------
# Index
# ---------------------------------------------------------------------------


class IndexRequest(BaseModel):
    """Incoming index request body."""

    directory: str = Field(
        ...,
        min_length=1,
        description="Path to a directory containing documents to index.",
        examples=["data/raw"],
    )


class IndexResponse(BaseModel):
    """Successful synchronous index response (legacy)."""

    files_processed: int = Field(0)
    chunks_created: int = Field(0)
    embeddings_generated: int = Field(0)


class IndexJobResponse(BaseModel):
    """Returned immediately when a background indexing job is created."""

    job_id: str = Field(..., description="UUID of the background job.")
    status: str = Field(default="QUEUED", description="Initial job status — always QUEUED.")


class JobStatusResponse(BaseModel):
    """Full status of a background indexing job."""

    job_id: str
    status: str
    created_at: str | None = None
    started_at: str | None = None
    completed_at: str | None = None
    files_processed: int = 0
    chunks_created: int = 0
    embeddings_generated: int = 0
    error_message: str | None = None


# ---------------------------------------------------------------------------
# Error
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------------


class UploadResponse(BaseModel):
    """Returned immediately after a file is accepted for upload."""

    filename: str = Field(..., description="Original filename as uploaded.")
    job_id: str = Field(..., description="Background indexing job triggered by this upload.")
    status: str = Field(default="QUEUED")


class UploadedFileInfo(BaseModel):
    """Metadata for a previously uploaded file."""

    filename: str
    size: int
    uploaded_at: str


# ---------------------------------------------------------------------------
# Error
# ---------------------------------------------------------------------------


class ErrorResponse(BaseModel):
    """Standard error envelope returned for all non-2xx responses."""

    detail: str = Field(..., description="Human-readable error description.")
    error_type: str = Field(..., description="Machine-readable error category.")
