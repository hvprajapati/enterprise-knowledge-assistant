import logging
import sqlite3

logger = logging.getLogger(__name__)


def create_schema(connection: sqlite3.Connection) -> None:
    """Creates or migrates the database schema for chunk metadata.

    Safe to call on an existing database — missing columns are added
    via ALTER TABLE, and indexes are created when absent.
    """

    # --- WAL mode: readers don't block writers -------------------------
    connection.execute("PRAGMA journal_mode=WAL")

    # --- main table ----------------------------------------------------
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS chunks (

            vector_id INTEGER PRIMARY KEY,

            chunk_id   TEXT NOT NULL,

            document_id TEXT NOT NULL,

            filename   TEXT    NOT NULL,
            extension  TEXT    NOT NULL,
            file_path  TEXT    NOT NULL,
            file_size  INTEGER NOT NULL,

            page_number INTEGER,

            chunk_index INTEGER NOT NULL,

            text TEXT NOT NULL,

            source        TEXT,
            document_type TEXT,
            tags          TEXT,

            content_hash TEXT

        );
        """
    )

    # --- migrations: add columns that may not exist in older DBs -------
    _add_column_if_missing(connection, "chunks", "content_hash", "TEXT")

    # --- indexes (perf-critical for query-time lookups) ----------------
    connection.execute(
        "CREATE INDEX IF NOT EXISTS idx_chunks_document_id "
        "ON chunks(document_id)"
    )
    connection.execute(
        "CREATE INDEX IF NOT EXISTS idx_chunks_filename "
        "ON chunks(filename)"
    )
    connection.execute(
        "CREATE INDEX IF NOT EXISTS idx_chunks_content_hash "
        "ON chunks(content_hash)"
    )

    connection.commit()


def _add_column_if_missing(
    connection: sqlite3.Connection,
    table: str,
    column: str,
    col_type: str,
) -> None:
    """Add *column* to *table* if it doesn't already exist.

    SQLite has no ``ADD COLUMN IF NOT EXISTS``, so we inspect
    ``PRAGMA table_info`` before running ``ALTER TABLE``.
    """
    existing = {
        row[1]
        for row in connection.execute(f"PRAGMA table_info({table})").fetchall()
    }
    if column not in existing:
        logger.info("Migrating schema: adding %s.%s %s", table, column, col_type)
        connection.execute(f"ALTER TABLE {table} ADD COLUMN {column} {col_type}")
