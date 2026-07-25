from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from pathlib import Path
from uuid import UUID

from app.ingestion.models import DocumentChunk, DocumentMetadata


class ChunkRepository:
    """Persists chunk metadata in SQLite, keyed by FAISS vector ID."""

    def __init__(self, connection: sqlite3.Connection) -> None:
        self.connection = connection

    # ------------------------------------------------------------------
    # write
    # ------------------------------------------------------------------

    def save_chunk(self, vector_id: int, chunk: DocumentChunk) -> None:
        """Insert a single chunk with its FAISS vector position."""
        self.connection.execute(
            """
            INSERT OR REPLACE INTO chunks (
                vector_id,
                chunk_id,
                document_id,
                filename,
                extension,
                file_path,
                file_size,
                page_number,
                chunk_index,
                text,
                source,
                document_type,
                tags,
                content_hash
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            _chunk_to_row(vector_id, chunk),
        )
        self.connection.commit()

    def save_chunks(self, entries: list[tuple[int, DocumentChunk]]) -> None:
        """Batch-insert chunks.  Much faster than calling ``save_chunk`` one at a time."""
        self.connection.executemany(
            """
            INSERT OR REPLACE INTO chunks (
                vector_id,
                chunk_id,
                document_id,
                filename,
                extension,
                file_path,
                file_size,
                page_number,
                chunk_index,
                text,
                source,
                document_type,
                tags,
                content_hash
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [_chunk_to_row(vid, chunk) for vid, chunk in entries],
        )
        self.connection.commit()

    # ------------------------------------------------------------------
    # read
    # ------------------------------------------------------------------

    def get_chunk(self, vector_id: int) -> DocumentChunk | None:
        """Return the chunk at *vector_id*, or ``None``."""
        row = self.connection.execute(
            "SELECT * FROM chunks WHERE vector_id = ?",
            (vector_id,),
        ).fetchone()

        if row is None:
            return None

        return _row_to_chunk(row)

    def get_chunks_by_document(
        self, document_id: UUID
    ) -> dict[int, DocumentChunk]:
        """Return every chunk for *document_id*, keyed by ``chunk_index``.

        Used by ``ParentDocumentRetriever`` to efficiently look up
        neighbouring chunks.
        """
        rows = self.connection.execute(
            "SELECT * FROM chunks WHERE document_id = ? ORDER BY chunk_index",
            (str(document_id),),
        ).fetchall()
        return {row["chunk_index"]: _row_to_chunk(row) for row in rows}

    def count(self) -> int:
        """Total number of chunks stored."""
        row = self.connection.execute(
            "SELECT COUNT(*) AS cnt FROM chunks"
        ).fetchone()
        return row["cnt"] if row else 0

    def iter_all(self) -> list[tuple[int, DocumentChunk]]:
        """Return every (vector_id, DocumentChunk) pair stored.

        Used at query time to hydrate the in-memory metadata store
        so ``FAISSVectorStore.search`` can resolve vector positions
        back to full chunk objects.

        .. note::
           For large corpora prefer ``iter_batches()`` to avoid
           loading everything into memory at once.
        """
        rows = self.connection.execute(
            "SELECT * FROM chunks ORDER BY vector_id"
        ).fetchall()

        return [(row["vector_id"], _row_to_chunk(row)) for row in rows]

    def iter_batches(
        self, batch_size: int = 1000
    ) -> Iterator[list[tuple[int, DocumentChunk]]]:
        """Yield chunks in batches to limit memory usage.

        Parameters
        ----------
        batch_size:
            Number of rows per batch (default 1000).

        Yields
        ------
        list[tuple[int, DocumentChunk]]
            One batch of (vector_id, chunk) pairs.
        """
        cursor = self.connection.execute(
            "SELECT * FROM chunks ORDER BY vector_id"
        )
        batch: list[tuple[int, DocumentChunk]] = []
        for row in cursor:
            batch.append((row["vector_id"], _row_to_chunk(row)))
            if len(batch) >= batch_size:
                yield batch
                batch = []
        if batch:
            yield batch

    # ------------------------------------------------------------------
    # duplicate detection
    # ------------------------------------------------------------------

    def has_content_hash(self, content_hash: str) -> bool:
        """Check whether a document with this SHA-256 hash already exists."""
        row = self.connection.execute(
            "SELECT 1 FROM chunks WHERE content_hash = ? LIMIT 1",
            (content_hash,),
        ).fetchone()
        return row is not None

    def get_content_hashes(self) -> set[str]:
        """Return the set of all known content hashes (for bulk dedup)."""
        rows = self.connection.execute(
            "SELECT DISTINCT content_hash FROM chunks WHERE content_hash IS NOT NULL"
        ).fetchall()
        return {r["content_hash"] for r in rows}


# ------------------------------------------------------------------
# internal helpers
# ------------------------------------------------------------------


def _chunk_to_row(vector_id: int, chunk: DocumentChunk) -> tuple:
    return (
        vector_id,
        str(chunk.chunk_id),
        str(chunk.document_id),
        chunk.metadata.filename,
        chunk.metadata.extension,
        str(chunk.metadata.file_path),
        chunk.metadata.file_size,
        chunk.page_number,
        chunk.chunk_index,
        chunk.text,
        chunk.metadata.source,
        chunk.metadata.document_type,
        ",".join(chunk.metadata.tags),
        getattr(chunk, "content_hash", None),
    )


def _row_to_chunk(row: sqlite3.Row) -> DocumentChunk:
    metadata = DocumentMetadata(
        document_id=UUID(row["document_id"]),
        filename=row["filename"],
        file_path=Path(row["file_path"]),
        extension=row["extension"],
        file_size=row["file_size"],
        source=row["source"],
        document_type=row["document_type"],
        tags=row["tags"].split(",") if row["tags"] else [],
    )

    return DocumentChunk(
        chunk_id=UUID(row["chunk_id"]),
        document_id=metadata.document_id,
        chunk_index=row["chunk_index"],
        page_number=row["page_number"],
        text=row["text"],
        metadata=metadata,
    )
