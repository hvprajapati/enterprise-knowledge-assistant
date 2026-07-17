"""Query orchestration layer.

Coordinates the full RAG pipeline::

    User Question
        |
    QueryRewriter         (expand / clarify — optional)
        |
    MultiQueryGenerator   (generate N diverse queries — optional)
        |
    EmbeddingService      (batch-encode all queries)
        |
    Retriever             (retrieve per query)
        |
    Merge + Deduplicate   (keep best score per chunk_id)
        |
    CrossEncoderReranker  (re-score merged passages)
        |
    PromptBuilder         (assemble RAG prompt)
        |
    BaseLLM               (generate answer)
        |
    QueryResponse         (structured result)
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING

from app.embeddings.service import EmbeddingService
from app.llm.base import BaseLLM
from app.llm.exceptions import LLMError
from app.llm.factory import LLMFactory
from app.prompts.builder import PromptBuilder
from app.query.models import QueryResponse
from app.retrieval.bm25 import BM25Retriever
from app.retrieval.context_compressor import BaseContextCompressor, SimilarityContextCompressor
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.multi_query import MultiQueryGenerator
from app.retrieval.query_rewriter import QueryRewriter
from app.retrieval.reranker import CrossEncoderReranker
from app.retrieval.retriever import Retriever
from app.storage.repository import ChunkRepository
from app.storage.sqlite import SQLiteConnection

if TYPE_CHECKING:
    from app.config.settings import Settings
    from app.ingestion.models import SearchResult

logger = logging.getLogger(__name__)


class QueryError(Exception):
    """Raised when the query pipeline encounters an unrecoverable error."""


class QueryService:
    """Orchestrates the end-to-end RAG pipeline.

    All dependencies are injected — the service owns no business logic
    beyond orchestration order.

    Parameters
    ----------
    embedding_service:
        Encodes the raw user question into a vector.
    retriever:
        Searches FAISS and returns metadata-hydrated results.
    reranker:
        Cross-encoder that re-scores retrieved passages.
    prompt_builder:
        Assembles the system prompt + context + question.
    llm:
        Vendor-independent LLM provider.
    """

    def __init__(
        self,
        embedding_service: EmbeddingService,
        retriever: Retriever,
        reranker: CrossEncoderReranker,
        prompt_builder: PromptBuilder,
        llm: BaseLLM,
        *,
        rewriter: QueryRewriter | None = None,
        multi_query: MultiQueryGenerator | None = None,
        hybrid_retriever: HybridRetriever | None = None,
        compressor: BaseContextCompressor | None = None,
    ) -> None:
        self._embed = embedding_service
        self._retriever = retriever
        self._hybrid = hybrid_retriever
        self._reranker = reranker
        self._prompt_builder = prompt_builder
        self._llm = llm
        self._rewriter = rewriter
        self._multi_query = multi_query
        self._compressor = compressor

    # ------------------------------------------------------------------
    # factory
    # ------------------------------------------------------------------

    @classmethod
    def from_paths(
        cls,
        index_path: Path,
        database_path: Path,
        settings: Settings | None = None,
    ) -> QueryService:
        """Build a ready-to-use ``QueryService`` from persisted artefacts.

        Parameters
        ----------
        index_path:
            Path to the saved FAISS index.
        database_path:
            Path to the SQLite metadata database.
        settings:
            Application settings.  Uses the module-level singleton when
            ``None``.
        """
        if settings is None:
            from app.config.settings import settings as _settings

            settings = _settings

        embedding_service = EmbeddingService()
        retriever = Retriever.from_paths(index_path, database_path)
        reranker = CrossEncoderReranker()
        prompt_builder = PromptBuilder()
        llm = LLMFactory(settings).create()
        rewriter = QueryRewriter(llm) if settings.query_rewriter_enabled else None
        multi_query = MultiQueryGenerator(llm) if settings.multi_query_enabled else None
        compressor = (
            SimilarityContextCompressor(embedding_service)
            if settings.enable_context_compression
            else None
        )

        # Build hybrid retriever (dense + BM25) when enabled
        hybrid_retriever: HybridRetriever | None = None
        if settings.enable_hybrid_search:
            sqlite = SQLiteConnection(database_path)
            conn = sqlite.connect()
            try:
                repo = ChunkRepository(conn)
                all_chunks = [chunk for _, chunk in repo.iter_all()]
                if all_chunks:
                    bm25 = BM25Retriever(all_chunks)
                    hybrid_retriever = HybridRetriever(
                        dense=retriever,
                        bm25=bm25,
                    )
                    logger.info("Hybrid retriever initialised — %d BM25 docs", len(all_chunks))
                else:
                    logger.warning("No chunks in database — hybrid search disabled")
            finally:
                conn.close()

        return cls(
            embedding_service=embedding_service,
            retriever=retriever,
            reranker=reranker,
            prompt_builder=prompt_builder,
            llm=llm,
            rewriter=rewriter,
            multi_query=multi_query,
            hybrid_retriever=hybrid_retriever,
            compressor=compressor,
        )

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def answer(
        self,
        question: str,
        *,
        top_k: int = 50,
        rerank_top_k: int = 5,
        source: str | None = None,
        document_type: str | None = None,
        tags: list[str] | None = None,
    ) -> QueryResponse:
        """Run the complete RAG pipeline and return a structured response.

        Parameters
        ----------
        question:
            The user's raw question.  Must be non-empty.
        top_k:
            Candidate passages to retrieve from FAISS (default 50).
        rerank_top_k:
            Passages kept after cross-encoder reranking (default 5).
        source:
            Optional metadata filter — keep only chunks matching this
            source value.
        document_type:
            Optional metadata filter — keep only chunks matching this
            document type.
        tags:
            Optional metadata filter — keep only chunks that have
            **all** of the supplied tags.

        Returns
        -------
        QueryResponse
            Contains the LLM answer, source list, token counts
            (when available), and total latency.

        Raises
        ------
        QueryError
            Wraps any failure from the pipeline stages.
        """
        t_start = time.monotonic()

        # -- 0. validate -------------------------------------------------
        question = question.strip()
        if not question:
            raise QueryError("question must be a non-empty string")

        logger.info("Query request started — question=%d chars", len(question))
        retrieved_count = 0
        reranked_count = 0

        try:
            # -- 0. rewrite ------------------------------------------------
            if self._rewriter is not None:
                question = self._rewriter.rewrite(question)

            # -- 1. multi-query generation + embed + retrieve + merge -----
            candidates, retrieved_count = self._retrieve_multi(
                question, top_k=top_k
            )

            # -- 2. rerank ------------------------------------------------
            t_rerank = time.monotonic()
            reranked: list[SearchResult] = self._reranker.rerank(
                question, candidates, top_k=rerank_top_k
            )
            reranked_count = len(reranked)
            logger.info(
                "Reranking completed — %d passages (rerank_top_k=%d  latency=%.0fms)",
                reranked_count,
                rerank_top_k,
                (time.monotonic() - t_rerank) * 1000,
            )

            # -- 3. compress ------------------------------------------------
            if self._compressor is not None:
                t_compress = time.monotonic()
                try:
                    reranked = self._compressor.compress(question, reranked)
                    logger.info(
                        "Compression completed — %d chunks  latency=%.0fms",
                        len(reranked),
                        (time.monotonic() - t_compress) * 1000,
                    )
                except Exception:
                    logger.exception(
                        "Context compression failed — continuing with original results"
                    )

            # -- 4. build prompt ------------------------------------------
            t_prompt = time.monotonic()
            prompt = self._prompt_builder.build_prompt(question, reranked)
            logger.info(
                "Prompt generated — %d chars  latency=%.0fms",
                len(prompt),
                (time.monotonic() - t_prompt) * 1000,
            )

            # -- 5. LLM ---------------------------------------------------
            t_llm = time.monotonic()
            answer = self._llm.generate(prompt)
            llm_latency = (time.monotonic() - t_llm) * 1000
            logger.info(
                "LLM completed — answer=%d chars  latency=%.0fms",
                len(answer),
                llm_latency,
            )

        except LLMError:
            raise
        except Exception as exc:
            raise QueryError(f"Query pipeline failed: {exc}") from exc

        # -- 6. assemble response ----------------------------------------
        latency_ms = (time.monotonic() - t_start) * 1000

        sources = list({
            r.chunk.metadata.filename
            for r in reranked
        })

        logger.info(
            "Query request finished — latency=%.0fms  retrieved=%d  reranked=%d",
            latency_ms,
            retrieved_count,
            reranked_count,
        )

        return QueryResponse(
            answer=answer,
            sources=sources,
            retrieved_count=retrieved_count,
            reranked_count=reranked_count,
            latency_ms=latency_ms,
        )

    # ------------------------------------------------------------------
    # multi-query retrieval helpers
    # ------------------------------------------------------------------

    def _retrieve_multi(
        self,
        question: str,
        *,
        top_k: int = 50,
    ) -> tuple[list[SearchResult], int]:
        """Run multi-query retrieval and return (merged_results, total_count)."""
        # -- generate queries --------------------------------------------
        queries: list[str]
        if self._multi_query is not None:
            queries = self._multi_query.generate(question)
        else:
            queries = [question]

        logger.info(
            "Multi-query: %d queries generated (enabled=%s)",
            len(queries),
            self._multi_query is not None,
        )

        # -- batch-embed all queries -------------------------------------
        t_embed = time.monotonic()
        embeddings = self._embed.embed_queries(queries)
        logger.debug(
            "Batch embedding — %d queries  dim=%d  latency=%.0fms",
            len(embeddings),
            len(embeddings[0]) if embeddings else 0,
            (time.monotonic() - t_embed) * 1000,
        )

        # -- retrieve per query ------------------------------------------
        t_retrieve = time.monotonic()
        results_per_query: list[list[SearchResult]] = []
        total_before_merge = 0
        for i, (emb, query_text) in enumerate(zip(embeddings, queries, strict=True)):
            if self._hybrid is not None:
                batch = self._hybrid.retrieve(emb, query_text, top_k=top_k)
            else:
                batch = self._retriever.retrieve(emb, top_k=top_k)
            results_per_query.append(batch)
            total_before_merge += len(batch)
            logger.debug("  query[%d] retrieved %d results", i, len(batch))

        # -- merge + deduplicate -----------------------------------------
        merged = self._merge_deduplicate(results_per_query)
        duplicates_removed = total_before_merge - len(merged)

        logger.info(
            "Multi-query retrieval — %d queries  %d raw  %d duplicates  %d merged  latency=%.0fms",
            len(queries),
            total_before_merge,
            duplicates_removed,
            len(merged),
            (time.monotonic() - t_retrieve) * 1000,
        )

        return merged, total_before_merge

    @staticmethod
    def _merge_deduplicate(
        results_per_query: list[list[SearchResult]],
    ) -> list[SearchResult]:
        """Merge multiple retrieval batches, keeping the highest score per chunk_id."""
        best: dict[str, SearchResult] = {}
        for batch in results_per_query:
            for result in batch:
                cid = str(result.chunk.chunk_id)
                if cid not in best or result.score > best[cid].score:
                    best[cid] = result
        return sorted(best.values(), key=lambda r: r.score, reverse=True)

    # ------------------------------------------------------------------
    # streaming
    # ------------------------------------------------------------------

    def stream_answer(
        self,
        question: str,
        *,
        top_k: int = 50,
        rerank_top_k: int = 5,
        source: str | None = None,
        document_type: str | None = None,
        tags: list[str] | None = None,
    ) -> Iterator[str]:
        """Run the RAG pipeline and yield LLM tokens as they arrive.

        The pipeline (embed → retrieve → rerank → prompt) is identical
        to ``answer``.  Only the final LLM call switches from
        ``generate`` to ``stream_generate``.

        Parameters
        ----------
        question:
            The user's raw question.  Must be non-empty.
        top_k:
            Candidate passages to retrieve from FAISS (default 50).
        rerank_top_k:
            Passages kept after cross-encoder reranking (default 5).
        source, document_type, tags:
            Optional metadata filters.

        Yields
        ------
        str
            Incremental text tokens from the LLM.

        Raises
        ------
        QueryError
            Wraps any failure from the pipeline stages (pre-LLM).
        LLMError
            Provider-level errors propagate directly.
        """
        # -- 0. validate -------------------------------------------------
        question = question.strip()
        if not question:
            raise QueryError("question must be a non-empty string")

        logger.info("Streaming query started — question=%d chars", len(question))

        try:
            # -- 0. rewrite ------------------------------------------------
            if self._rewriter is not None:
                question = self._rewriter.rewrite(question)

            # -- 1. multi-query generation + embed + retrieve + merge -----
            candidates, _ = self._retrieve_multi(question, top_k=top_k)

            # -- 2. rerank ------------------------------------------------
            reranked: list[SearchResult] = self._reranker.rerank(
                question, candidates, top_k=rerank_top_k
            )
            logger.info(
                "Stream reranking — %d passages (rerank_top_k=%d)",
                len(reranked),
                rerank_top_k,
            )

            # -- 3. compress ------------------------------------------------
            if self._compressor is not None:
                try:
                    reranked = self._compressor.compress(question, reranked)
                except Exception:
                    logger.exception("Stream compression failed — continuing")

            # -- 4. build prompt ------------------------------------------
            prompt = self._prompt_builder.build_prompt(question, reranked)
            logger.info("Stream prompt — %d chars", len(prompt))

            # -- 5. stream LLM --------------------------------------------
            t_llm = time.monotonic()
            token_count = 0

            for token in self._llm.stream_generate(prompt):
                token_count += 1
                yield token

            llm_latency = (time.monotonic() - t_llm) * 1000
            logger.info(
                "Stream LLM completed — chunks=%d latency=%.0fms",
                token_count,
                llm_latency,
            )

        except LLMError:
            raise
        except Exception as exc:
            raise QueryError(f"Query pipeline failed: {exc}") from exc
