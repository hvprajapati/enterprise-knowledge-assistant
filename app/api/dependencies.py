"""FastAPI dependency injection.

Every dependency is a callable that FastAPI invokes per-request.
No business logic lives here — only wiring.

QueryService is cached at module level so FAISS, BM25, and the
cross-encoder are loaded once at startup rather than on every request.
Call ``invalidate_query_service_cache()`` after re-indexing.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path

from app.config.settings import settings
from app.indexing.index_builder import IndexBuilder
from app.jobs.manager import JobManager, job_manager
from app.query.service import QueryService
from app.storage.repository import ChunkRepository
from app.storage.schema import create_schema
from app.storage.sqlite import SQLiteConnection

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# QueryService — singleton cache (issue #41 / #50)
# ------------------------------------------------------------------

_query_service_lock = threading.Lock()
_query_service: QueryService | None = None


def get_query_service() -> QueryService:
    """Return a cached ``QueryService`` loaded from disk.

    The instance is created once and reused across requests.
    FAISS, BM25, and the cross-encoder model stay resident in memory.
    """
    global _query_service

    if _query_service is not None:
        return _query_service

    with _query_service_lock:
        # Double-check inside the lock
        if _query_service is not None:
            return _query_service

        logger.info("Loading QueryService — this may take a few seconds...")
        _query_service = QueryService.from_paths(
            index_path=Path(settings.index_storage_path),
            database_path=Path(settings.database_storage_path),
        )
        logger.info("QueryService cached — ready for requests")

    return _query_service


def invalidate_query_service_cache() -> None:
    """Drop the cached QueryService so the next request reloads.

    Call this after re-indexing so queries see the updated FAISS
    index and BM25 corpus.
    """
    global _query_service
    from app.retrieval.bm25 import invalidate_bm25_cache

    with _query_service_lock:
        _query_service = None
        logger.info("QueryService cache invalidated")

    invalidate_bm25_cache()


# ------------------------------------------------------------------
# Index Builder
# ------------------------------------------------------------------


def get_index_builder() -> IndexBuilder:
    """Return an ``IndexBuilder`` wired to the configured SQLite database."""
    db_path = Path(settings.database_storage_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    sqlite = SQLiteConnection(db_path)
    connection = sqlite.connect()
    create_schema(connection)

    repository = ChunkRepository(connection)
    return IndexBuilder(
        repository,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        min_chunk_size=settings.min_chunk_size,
    )


def build_index_builder() -> IndexBuilder:
    """Create a fresh ``IndexBuilder`` wired to the configured database.

    Used by background tasks which run outside FastAPI's DI scope.
    """
    db_path = Path(settings.database_storage_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    sqlite = SQLiteConnection(db_path)
    connection = sqlite.connect()
    create_schema(connection)

    repository = ChunkRepository(connection)
    return IndexBuilder(
        repository,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        min_chunk_size=settings.min_chunk_size,
    )


# ------------------------------------------------------------------
# Job Manager
# ------------------------------------------------------------------


def get_job_manager() -> JobManager:
    """Return the process-level singleton ``JobManager``."""
    return job_manager
