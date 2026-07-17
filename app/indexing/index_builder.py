from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import faiss

from app.embeddings.service import EmbeddingService
from app.ingestion.ingestion_service import IngestionService
from app.ingestion.models import DocumentChunk
from app.storage.repository import ChunkRepository
from app.vectorstore.faiss_store import FAISSVectorStore

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS: set[str] = {".pdf", ".docx", ".txt", ".md"}


@dataclass
class IndexingStats:
    files_processed: int = 0
    chunks_created: int = 0
    embeddings_generated: int = 0


class IndexBuilder:
    """Builds the FAISS vector index from a directory of documents."""

    def __init__(
        self,
        repository: ChunkRepository,
        *,
        batch_size: int = 32,
    ) -> None:
        self.ingestion = IngestionService()
        self.embedding = EmbeddingService()
        self.repository = repository
        self.batch_size = batch_size

        self.vector_store = FAISSVectorStore(
            dimension=self.embedding.dimension,
        )

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def build(
        self,
        input_folder: Path,
        index_path: Path,
    ) -> IndexingStats:
        """Run the full indexing pipeline.

        Returns
        -------
        IndexingStats
            Counts for files, chunks, and embeddings processed.
        """
        stats = IndexingStats()

        chunk_buffer: list[DocumentChunk] = []

        for file_path in self._discover_files(input_folder):
            try:
                chunks = self.ingestion.ingest(file_path)
            except Exception:
                logger.exception("Failed to ingest: %s", file_path)
                continue

            stats.files_processed += 1
            stats.chunks_created += len(chunks)
            chunk_buffer.extend(chunks)

            # Flush buffer when it reaches batch_size
            while len(chunk_buffer) >= self.batch_size:
                batch = chunk_buffer[: self.batch_size]
                del chunk_buffer[: self.batch_size]
                stats.embeddings_generated += self._embed_and_store(batch)

        # Flush any remaining chunks
        if chunk_buffer:
            stats.embeddings_generated += self._embed_and_store(chunk_buffer)

        self._save_index(index_path)

        logger.info(
            "Indexing complete — files: %d  chunks: %d  embeddings: %d",
            stats.files_processed,
            stats.chunks_created,
            stats.embeddings_generated,
        )

        return stats

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------

    def _discover_files(self, root: Path) -> list[Path]:
        """Return supported files under *root* in a stable order."""
        files: list[Path] = []
        for path in sorted(root.rglob("*")):
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
                files.append(path)
        return files

    def _embed_and_store(self, chunks: list[DocumentChunk]) -> int:
        """Embed a batch of chunks, store vectors + metadata.

        Returns the number of embeddings successfully stored.
        """
        if not chunks:
            return 0

        embeddings = self.embedding.embed_chunks(chunks)

        entries: list[tuple[int, DocumentChunk]] = []
        stored = 0

        for chunk, embedding in zip(chunks, embeddings, strict=True):
            vector_id = self.vector_store.add(embedding, chunk)
            entries.append((vector_id, chunk))
            stored += 1

        self.repository.save_chunks(entries)

        logger.debug("Embedded and stored %d chunks (batch)", stored)
        return stored

    def _save_index(self, index_path: Path) -> None:
        """Persist the FAISS index to disk."""
        index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.vector_store.index, str(index_path))
        logger.info(
            "FAISS index saved to %s (%d vectors)", index_path, self.vector_store.index.ntotal
        )
