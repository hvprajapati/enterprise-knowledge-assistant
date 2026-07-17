"""Retrieval layer — FAISS search + SQLite metadata hydration.

This module is deliberately narrow.  It does **not** generate embeddings,
build prompts, or call any LLM.  Those responsibilities belong to other
layers.
"""

from __future__ import annotations

import logging
from pathlib import Path

import faiss

from app.ingestion.models import SearchResult
from app.storage.repository import ChunkRepository
from app.storage.schema import create_schema
from app.storage.sqlite import SQLiteConnection
from app.vectorstore.faiss_store import FAISSVectorStore

logger = logging.getLogger(__name__)


class Retriever:
    """Search a FAISS index and return metadata-enriched results.

    Dependencies are injected via the constructor.  Use the
    ``from_paths`` class-method factory for the common case of loading a
    persisted index and SQLite database from disk.

    Parameters
    ----------
    vector_store:
        FAISS index wrapper whose internal ``metadata_store`` has
        already been hydrated with chunks.
    chunk_repository:
        SQLite-backed repository — used during initialisation to hydrate
        the metadata store and kept for any future lookup needs.
    """

    def __init__(
        self,
        vector_store: FAISSVectorStore,
        chunk_repository: ChunkRepository,
    ) -> None:
        self._store = vector_store
        self._repo = chunk_repository

    # ------------------------------------------------------------------
    # factory
    # ------------------------------------------------------------------

    @classmethod
    def from_paths(
        cls,
        index_path: Path,
        database_path: Path,
    ) -> Retriever:
        """Create a ready-to-use ``Retriever`` from persisted artefacts.

        Raises
        ------
        FileNotFoundError
            If *index_path* or *database_path* does not exist.
        RuntimeError
            If the FAISS index is empty (0 vectors).
        """
        if not index_path.is_file():
            raise FileNotFoundError(f"FAISS index not found: {index_path}")
        if not database_path.is_file():
            raise FileNotFoundError(
                f"Metadata database not found: {database_path}"
            )

        # -- load FAISS index ------------------------------------------
        index = faiss.read_index(str(index_path))

        if index.ntotal == 0:
            raise RuntimeError(f"FAISS index is empty: {index_path}")

        dimension = index.d
        vector_store = FAISSVectorStore(dimension=dimension)
        vector_store.index = index

        logger.info(
            "Loaded FAISS index from %s (%d vectors, dim=%d)",
            index_path,
            index.ntotal,
            dimension,
        )

        # -- hydrate metadata store from SQLite ------------------------
        sqlite = SQLiteConnection(database_path)
        connection = sqlite.connect()

        try:
            create_schema(connection)
            repository = ChunkRepository(connection)

            entries = repository.iter_all()
            for vector_id, chunk in entries:
                vector_store.metadata_store.add(vector_id, chunk)

            logger.info(
                "Hydrated %d metadata records from %s",
                len(entries),
                database_path,
            )
        except Exception:
            connection.close()
            raise

        return cls(
            vector_store=vector_store,
            chunk_repository=repository,
        )

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query_embedding: list[float],
        top_k: int = 20,
    ) -> list[SearchResult]:
        """Return the top-*k* results for a pre-computed query embedding.

        The caller is responsible for generating the embedding
        (e.g. via ``EmbeddingService.embed_query``).

        Parameters
        ----------
        query_embedding:
            Normalised embedding vector of the user query.
        top_k:
            Maximum number of results to return (default 20).

        Returns
        -------
        list[SearchResult]
            Results ordered by descending inner-product similarity.
            May be empty if the index has fewer vectors than *top_k*
            or if all candidate chunks are orphaned.
        """
        return self._store.search(query_embedding, top_k=top_k)
