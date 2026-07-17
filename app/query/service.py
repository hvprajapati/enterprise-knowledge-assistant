"""Query orchestration layer.

Coordinates the full RAG pipeline::

    User Question
        |
    QueryRewriter      (expand / clarify — optional, configurable)
        |
    EmbeddingService   (text → vector)
        |
    Retriever          (FAISS search + metadata hydration)
        |
    CrossEncoderReranker  (re-score passages)
        |
    PromptBuilder      (assemble RAG prompt)
        |
    BaseLLM            (generate answer)
        |
    QueryResponse      (structured result)
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
from app.retrieval.query_rewriter import QueryRewriter
from app.retrieval.reranker import CrossEncoderReranker
from app.retrieval.retriever import Retriever

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
    ) -> None:
        self._embed = embedding_service
        self._retriever = retriever
        self._reranker = reranker
        self._prompt_builder = prompt_builder
        self._llm = llm
        self._rewriter = rewriter

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

        return cls(
            embedding_service=embedding_service,
            retriever=retriever,
            reranker=reranker,
            prompt_builder=prompt_builder,
            llm=llm,
            rewriter=rewriter,
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

            # -- 1. embed -------------------------------------------------
            t_embed = time.monotonic()
            embedding = self._embed.embed_query(question)
            logger.debug(
                "Embedding generated — dim=%d latency=%.0fms",
                len(embedding),
                (time.monotonic() - t_embed) * 1000,
            )

            # -- 2. retrieve ----------------------------------------------
            t_retrieve = time.monotonic()
            candidates: list[SearchResult] = self._retriever.retrieve(
                embedding, top_k=top_k
            )
            retrieved_count = len(candidates)
            logger.info(
                "Retrieval completed — %d candidates (top_k=%d  latency=%.0fms)",
                retrieved_count,
                top_k,
                (time.monotonic() - t_retrieve) * 1000,
            )

            # -- 3. rerank ------------------------------------------------
            t_rerank = time.monotonic()
            reranked: list[SearchResult] = self._reranker.rerank(
                question,
                candidates,
                top_k=rerank_top_k,
            )
            reranked_count = len(reranked)
            logger.info(
                "Reranking completed — %d passages (rerank_top_k=%d  latency=%.0fms)",
                reranked_count,
                rerank_top_k,
                (time.monotonic() - t_rerank) * 1000,
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

            # -- 1. embed -------------------------------------------------
            embedding = self._embed.embed_query(question)

            # -- 2. retrieve ----------------------------------------------
            candidates: list[SearchResult] = self._retriever.retrieve(
                embedding, top_k=top_k
            )
            logger.info(
                "Stream retrieval — %d candidates (top_k=%d)",
                len(candidates),
                top_k,
            )

            # -- 3. rerank ------------------------------------------------
            reranked: list[SearchResult] = self._reranker.rerank(
                question, candidates, top_k=rerank_top_k
            )
            logger.info(
                "Stream reranking — %d passages (rerank_top_k=%d)",
                len(reranked),
                rerank_top_k,
            )

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
