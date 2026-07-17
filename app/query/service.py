"""Query orchestration layer.

Uses a ``RetrievalOrchestrator`` to build a ``RetrievalPlan``
tailored to the user's question.  Only the stages enabled in the
plan are executed — minimising latency and LLM cost.
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
from app.retrieval.orchestrator.orchestrator import RetrievalOrchestrator
from app.retrieval.parent_document import ParentDocumentRetriever
from app.retrieval.query_rewriter import QueryRewriter
from app.retrieval.reranker import CrossEncoderReranker
from app.retrieval.retriever import Retriever
from app.retrieval.self_query.parser import SelfQueryParser
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
        parent_retriever: ParentDocumentRetriever | None = None,
        self_query: SelfQueryParser | None = None,
        orchestrator: RetrievalOrchestrator | None = None,
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
        self._parent = parent_retriever
        self._self_query = self_query
        self._orchestrator = orchestrator or RetrievalOrchestrator()

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
        self_query = SelfQueryParser(llm) if settings.enable_self_query else None
        multi_query = MultiQueryGenerator(llm) if settings.multi_query_enabled else None
        compressor = (
            SimilarityContextCompressor(embedding_service)
            if settings.enable_context_compression
            else None
        )

        # Parent document expansion
        parent_retriever: ParentDocumentRetriever | None = None
        if settings.enable_parent_document_retrieval:
            sqlite_p = SQLiteConnection(database_path)
            conn_p = sqlite_p.connect()
            try:
                repo_p = ChunkRepository(conn_p)
                parent_retriever = ParentDocumentRetriever(repo_p)
                logger.info("Parent document retriever initialised")
            finally:
                conn_p.close()

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
            parent_retriever=parent_retriever,
            self_query=self_query,
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

        # -- 1. plan -----------------------------------------------------
        plan = self._orchestrator.plan(question)
        logger.info(
            "Plan: type=%s  stages=%s  vector_k=%d  bm25_k=%d",
            plan.question_type.value,
            plan.stages_enabled,
            plan.vector_top_k,
            plan.bm25_top_k,
        )

        retrieved_count = 0
        reranked_count = 0

        try:
            # -- 2. rewrite (plan-controlled) ----------------------------
            if plan.rewrite_query and self._rewriter is not None:
                question = self._rewriter.rewrite(question)

            # -- 3. self-query (plan-controlled) -------------------------
            metadata_filters: dict[str, str | None] | None = None
            if plan.use_self_query and self._self_query is not None:
                parsed = self._self_query.parse(question)
                if parsed.metadata_filters:
                    metadata_filters = parsed.metadata_filters
                question = parsed.rewritten_query or question

            # -- 4. retrieve (plan-controlled multi-query + hybrid) ------
            use_hybrid = plan.use_hybrid_search and self._hybrid is not None
            use_multi = plan.use_multi_query and self._multi_query is not None

            if use_multi:
                queries = self._multi_query.generate(question)  # type: ignore[union-attr]
            else:
                queries = [question]

            logger.info(
                "Retrieval: %d queries  hybrid=%s  filters=%s",
                len(queries), use_hybrid, bool(metadata_filters),
            )

            embeddings = self._embed.embed_queries(queries)
            results_per_query: list[list[SearchResult]] = []
            total_before_merge = 0

            for emb, qtext in zip(embeddings, queries, strict=True):
                if use_hybrid:
                    batch = self._hybrid.retrieve(  # type: ignore[union-attr]
                        emb, qtext,
                        top_k=plan.vector_top_k,
                        metadata_filters=metadata_filters,
                    )
                else:
                    batch = self._retriever.retrieve(emb, top_k=plan.vector_top_k)
                results_per_query.append(batch)
                total_before_merge += len(batch)

            candidates = self._merge_deduplicate(results_per_query)
            retrieved_count = total_before_merge
            logger.info(
                "Retrieval: %d queries -> %d raw -> %d unique",
                len(queries), total_before_merge, len(candidates),
            )

            # -- 5. rerank (always on) -----------------------------------
            if plan.rerank:
                t_rerank = time.monotonic()
                reranked = self._reranker.rerank(question, candidates, top_k=rerank_top_k)
                reranked_count = len(reranked)
                logger.info(
                    "Reranked: %d -> %d (%.0fms)",
                    len(candidates), reranked_count,
                    (time.monotonic() - t_rerank) * 1000,
                )
            else:
                reranked = candidates[:rerank_top_k]
                reranked_count = len(reranked)

            # -- 6. parent expansion (plan-controlled) -------------------
            if plan.use_parent_document and self._parent is not None:
                try:
                    reranked = self._parent.expand(reranked)
                    logger.info("Parent expansion: %d chunks", len(reranked))
                except Exception:
                    logger.exception("Parent expansion failed — continuing")

            # -- 7. compress (plan-controlled) ---------------------------
            if plan.use_context_compression and self._compressor is not None:
                try:
                    reranked = self._compressor.compress(question, reranked)
                    logger.info("Compression: %d chunks", len(reranked))
                except Exception:
                    logger.exception("Compression failed — continuing")

            # -- 8. build prompt -----------------------------------------
            t_prompt = time.monotonic()
            prompt = self._prompt_builder.build_prompt(question, reranked)
            logger.info(
                "Prompt: %d chars (%.0fms)",
                len(prompt), (time.monotonic() - t_prompt) * 1000,
            )

            # -- 9. LLM --------------------------------------------------
            t_llm = time.monotonic()
            answer = self._llm.generate(prompt)
            logger.info(
                "LLM: %d chars (%.0fms)",
                len(answer), (time.monotonic() - t_llm) * 1000,
            )

        except LLMError:
            raise
        except Exception as exc:
            raise QueryError(f"Query pipeline failed: {exc}") from exc

        # -- 10. assemble response ---------------------------------------
        latency_ms = (time.monotonic() - t_start) * 1000

        sources = list({r.chunk.metadata.filename for r in reranked})

        logger.info(
            "Query finished — latency=%.0fms  retrieved=%d  reranked=%d",
            latency_ms, retrieved_count, reranked_count,
        )

        return QueryResponse(
            answer=answer,
            sources=sources,
            retrieved_count=retrieved_count,
            reranked_count=reranked_count,
            latency_ms=latency_ms,
        )

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

        # -- plan ---------------------------------------------------------
        plan = self._orchestrator.plan(question)

        try:
            # -- rewrite (plan-controlled) --------------------------------
            if plan.rewrite_query and self._rewriter is not None:
                question = self._rewriter.rewrite(question)

            # -- self-query (plan-controlled) -----------------------------
            metadata_filters: dict[str, str | None] | None = None
            if plan.use_self_query and self._self_query is not None:
                parsed = self._self_query.parse(question)
                if parsed.metadata_filters:
                    metadata_filters = parsed.metadata_filters
                question = parsed.rewritten_query or question

            # -- retrieve (plan-controlled) -------------------------------
            use_hybrid = plan.use_hybrid_search and self._hybrid is not None
            use_multi = plan.use_multi_query and self._multi_query is not None

            queries = (
                self._multi_query.generate(question)  # type: ignore[union-attr]
                if use_multi
                else [question]
            )
            embeddings = self._embed.embed_queries(queries)
            results_per_query: list[list[SearchResult]] = []

            for emb, qtext in zip(embeddings, queries, strict=True):
                if use_hybrid:
                    batch = self._hybrid.retrieve(  # type: ignore[union-attr]
                        emb, qtext, top_k=plan.vector_top_k,
                        metadata_filters=metadata_filters,
                    )
                else:
                    batch = self._retriever.retrieve(emb, top_k=plan.vector_top_k)
                results_per_query.append(batch)

            candidates = self._merge_deduplicate(results_per_query)

            # -- rerank ---------------------------------------------------
            reranked = self._reranker.rerank(question, candidates, top_k=rerank_top_k)
            logger.info("Stream reranking — %d passages", len(reranked))

            # -- parent expansion (plan-controlled) -----------------------
            if plan.use_parent_document and self._parent is not None:
                try:
                    reranked = self._parent.expand(reranked)
                except Exception:
                    logger.exception("Stream parent expansion failed — continuing")

            # -- compress (plan-controlled) -------------------------------
            if plan.use_context_compression and self._compressor is not None:
                try:
                    reranked = self._compressor.compress(question, reranked)
                except Exception:
                    logger.exception("Stream compression failed — continuing")

            # -- build prompt ---------------------------------------------
            prompt = self._prompt_builder.build_prompt(question, reranked)
            logger.info("Stream prompt — %d chars", len(prompt))

            # -- stream LLM -----------------------------------------------
            t_llm = time.monotonic()
            token_count = 0

            for token in self._llm.stream_generate(prompt):
                token_count += 1
                yield token

            logger.info(
                "Stream LLM completed — chunks=%d latency=%.0fms",
                token_count, (time.monotonic() - t_llm) * 1000,
            )

        except LLMError:
            raise
        except Exception as exc:
            raise QueryError(f"Query pipeline failed: {exc}") from exc
