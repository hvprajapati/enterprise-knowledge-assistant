"""FastAPI dependency injection.

Every dependency is a callable that FastAPI invokes per-request.
No business logic lives here — only wiring.
"""

from __future__ import annotations

from pathlib import Path

from app.config.settings import settings
from app.indexing.index_builder import IndexBuilder
from app.jobs.manager import JobManager, job_manager
from app.query.service import QueryService
from app.storage.repository import ChunkRepository
from app.storage.schema import create_schema
from app.storage.sqlite import SQLiteConnection


def get_query_service() -> QueryService:
    """Return a ready-to-use ``QueryService`` loaded from disk."""
    return QueryService.from_paths(
        index_path=Path(settings.index_storage_path),
        database_path=Path(settings.database_storage_path),
    )


def get_index_builder() -> IndexBuilder:
    """Return an ``IndexBuilder`` wired to the configured SQLite database."""
    db_path = Path(settings.database_storage_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    sqlite = SQLiteConnection(db_path)
    connection = sqlite.connect()
    create_schema(connection)

    repository = ChunkRepository(connection)
    return IndexBuilder(repository)


def get_job_manager() -> JobManager:
    """Return the process-level singleton ``JobManager``."""
    return job_manager


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
    return IndexBuilder(repository)
