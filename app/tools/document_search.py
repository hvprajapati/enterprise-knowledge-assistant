"""Document search tool — wraps the existing RAG QueryService.

This tool lets the agent explicitly search the document store from
within the tool-calling framework, separate from the automatic
retrieval pipeline.

Use cases:
- The planner identifies a metadata-filtered search is needed.
- The user explicitly asks to "search for X in the documents".
- Post-retrieval, a follow-up search is needed for clarification.
"""

from __future__ import annotations

from typing import Any

from app.tools.base import BaseTool


class DocumentSearchTool(BaseTool):
    """Search the enterprise document store using the RAG pipeline.

    This tool wraps the existing ``QueryService`` so the agent can
    perform explicit document searches when the automatic retrieval
    pipeline isn't sufficient (e.g. metadata-filtered queries).
    """

    def __init__(self) -> None:
        self._query_service: Any = None  # set via configure()

    # ------------------------------------------------------------------
    # configuration
    # ------------------------------------------------------------------

    def configure(self, query_service: Any) -> None:
        """Inject the QueryService dependency.

        Called during agent bootstrap.  The tool is registered first,
        then configured once the QueryService is available.
        """
        self._query_service = query_service

    # ------------------------------------------------------------------
    # BaseTool interface
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "document-search"

    @property
    def description(self) -> str:
        return (
            "Search the enterprise document store for information. "
            "Takes a search query string and returns relevant document "
            "passages. Use when the user wants to find information "
            "in uploaded documents, policies, reports, or manuals."
        )

    @property
    def schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query to find relevant documents.",
                },
                "top_k": {
                    "type": "integer",
                    "description": "Maximum number of results to return (default 5).",
                },
            },
            "required": ["query"],
        }

    def invoke(self, **kwargs: Any) -> dict[str, Any]:
        if self._query_service is None:
            raise RuntimeError(
                "DocumentSearchTool is not configured. "
                "Call configure(query_service) before invoking."
            )

        query = str(kwargs.get("query", ""))
        top_k = int(kwargs.get("top_k", 5))

        if not query.strip():
            raise ValueError("Search query is empty.")

        # Use the QueryService to answer (which internally runs the RAG pipeline)
        response = self._query_service.answer(query)

        # Extract top passages from the answer's sources
        sources = response.get("sources", []) if isinstance(response, dict) else []
        passages = [
            {
                "text": s.get("text", ""),
                "filename": s.get("filename", ""),
                "score": s.get("score", 0),
            }
            for s in sources[:top_k]
        ]

        return {
            "result": response.get("answer", "") if isinstance(response, dict) else str(response),
            "passages": passages,
            "total_sources": len(sources),
            "query": query,
        }
