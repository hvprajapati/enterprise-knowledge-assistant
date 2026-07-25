import sqlite3


def create_schema(connection: sqlite3.Connection) -> None:
    """Creates the database schema for chunk metadata.

    Includes performance-critical indexes and WAL journal mode
    for concurrent read/write safety.
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
