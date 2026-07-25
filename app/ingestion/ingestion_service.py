from __future__ import annotations

import hashlib
import logging
from pathlib import Path

from app.ingestion.chunker import DocumentChunker
from app.ingestion.cleaner import TextCleaner
from app.ingestion.models import DocumentChunk, DocumentPage, ParsedDocument
from app.ingestion.parser_factory import ParserFactory

logger = logging.getLogger(__name__)


class IngestionService:
    """Coordinates the document ingestion pipeline.

    Parameters
    ----------
    chunk_size:
        Maximum characters per chunk (default 800).
    chunk_overlap:
        Overlap between adjacent chunks (default 100).
    min_chunk_size:
        Minimum characters for a chunk to be kept (default 10).
    """

    def __init__(
        self,
        chunk_size: int = 800,
        chunk_overlap: int = 100,
        min_chunk_size: int = 10,
    ) -> None:
        self.cleaner = TextCleaner()
        self.chunker = DocumentChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            min_chunk_size=min_chunk_size,
        )

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def ingest(self, file_path: Path) -> list[DocumentChunk]:
        """Parse, clean, and chunk a document.

        A SHA-256 content hash is computed once per file and attached
        to every chunk so the indexer can detect duplicates.
        """
        content_hash = self.compute_content_hash(file_path)

        parser = ParserFactory.get_parser(file_path)

        document = parser.parse(file_path)

        cleaned_pages = [
            DocumentPage(
                page_number=page.page_number,
                text=self.cleaner.clean(page.text),
            )
            for page in document.pages
        ]

        cleaned_document = ParsedDocument(
            pages=cleaned_pages,
            metadata=document.metadata,
        )

        chunks = self.chunker.chunk(cleaned_document)

        # Attach content hash to every chunk for dedup
        for c in chunks:
            c.content_hash = content_hash

        logger.debug(
            "Ingested %s — %d chunks  hash=%s",
            file_path.name, len(chunks), content_hash[:12],
        )

        return chunks

    @staticmethod
    def compute_content_hash(file_path: Path) -> str:
        """Return the SHA-256 hex digest of a file's contents.

        Used for duplicate detection — if the same file content is
        uploaded twice (even under a different name), the hash will
        match and the second upload can be skipped.
        """
        sha = hashlib.sha256()
        with open(file_path, "rb") as fh:
            for chunk in iter(lambda: fh.read(65536), b""):
                sha.update(chunk)
        return sha.hexdigest()
