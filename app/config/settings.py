from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    project_name: str = "Enterprise Knowledge Assistant"
    version: str = "0.1.0"
    api_prefix: str = "/api/v1"
    environment: str = "development"

    reranker_model: str = "BAAI/bge-reranker-base"

    # -- prompt builder -------------------------------------------------
    prompt_system_text: str = (
        "You are an Enterprise Knowledge Assistant.\n\n"
        "Rules:\n"
        "  1. Answer ONLY using information present in the provided context.\n"
        "  2. Never fabricate, guess, or use external knowledge.\n"
        "  3. If the context does not contain the answer, state:\n"
        '     "I could not find the answer in the provided documents."\n'
        "  4. Cite sources using the [N] reference number shown above each passage.\n"
        "  5. Prefer concrete facts over vague summaries."
    )
    prompt_no_context_text: str = (
        "You are an Enterprise Knowledge Assistant.\n\n"
        "No documents were found matching the user's query. "
        "Inform the user that no relevant information is available "
        "and suggest they try a different question or upload additional documents."
    )
    prompt_max_context_chunks: int = 20

    # -- query rewriting -------------------------------------------------
    query_rewriter_enabled: bool = True
    query_rewriter_system_prompt: str = (
        "You are a query rewriter for a RAG system. "
        "Your job is to rewrite the user's question to improve retrieval quality.\n\n"
        "Rules:\n"
        "  1. NEVER answer the question. Only rewrite it.\n"
        "  2. Preserve the original intent exactly.\n"
        '  3. Expand abbreviations (e.g. "RAG" → "Retrieval-Augmented Generation").\n'
        "  4. Add missing context if it is obvious from the question itself.\n"
        '  5. Replace vague references (e.g. "it", "this", "that") with specific nouns.\n'
        "  6. Keep the rewritten question concise — no more than one sentence.\n"
        "  7. Output ONLY the rewritten question. No preamble, no explanation, no markdown.\n\n"
        "Examples:\n"
        '  Input:  "How does it work?"\n'
        '  Output: "How does Amazon Bedrock Knowledge Base retrieve and rank documents?"\n\n'
        '  Input:  "Explain RAG"\n'
        '  Output: "Explain Retrieval-Augmented Generation (RAG), including its indexing, '
        'retrieval, reranking, and LLM generation components."\n\n'
        '  Input:  "What are the benefits?"\n'
        '  Output: "What are the benefits of using a RAG system '
        'for enterprise knowledge management?"'
    )

    # -- multi-query retrieval ------------------------------------------
    multi_query_enabled: bool = True
    multi_query_max_variants: int = 4

    # -- hybrid retrieval ------------------------------------------------
    enable_hybrid_search: bool = True
    bm25_top_k: int = 50
    vector_top_k: int = 50
    rrf_k: int = 60

    # -- context compression ---------------------------------------------
    enable_context_compression: bool = True
    max_context_tokens: int = 4096
    redundancy_threshold: float = 0.92

    # -- parent document retrieval ---------------------------------------
    enable_parent_document_retrieval: bool = True
    parent_window_size: int = 1
    max_parent_chunks: int = 30

    # -- self-query retrieval --------------------------------------------
    enable_self_query: bool = True
    supported_metadata_fields: str = "filename,source,document_type,tags,extension"

    # -- validation thresholds -------------------------------------------
    validation_min_confidence_score: float = 0.6
    validation_require_grounded: bool = True
    validation_require_completeness: bool = True
    validation_require_relevance: bool = True

    # -- retry -----------------------------------------------------------
    max_agent_retries: int = 2

    # -- MCP server ------------------------------------------------------
    enable_mcp: bool = True
    mcp_server_name: str = "Enterprise Knowledge Assistant"
    mcp_server_version: str = "0.1.0"

    # -- MCP client ------------------------------------------------------
    enable_mcp_client: bool = False
    mcp_servers: str = "[]"  # JSON list of {name, command, args}
    auto_reconnect: bool = True
    heartbeat_interval: int = 30

    # -- evaluation ------------------------------------------------------
    enable_evaluation: bool = True
    ragas_thresholds: str = (
        '{"faithfulness":0.7,"answer_relevancy":0.6,'
        '"context_precision":0.6,"context_recall":0.6}'
    )
    deepeval_thresholds: str = (
        '{"hallucination":0.7,"faithfulness_de":0.7,'
        '"answer_relevancy_de":0.6,"bias":0.8,"toxicity":0.9}'
    )
    report_output_directory: str = "reports"

    multi_query_system_prompt: str = (
        "You are a search query generator for a RAG system. "
        "Given a user question, generate multiple semantically diverse "
        "search queries that will help retrieve relevant documents.\n\n"
        "Rules:\n"
        "  1. Generate exactly one query per line.\n"
        "  2. Each query must preserve the original user intent.\n"
        "  3. Use different wording, synonyms, and perspectives.\n"
        "  4. Include related technical terminology.\n"
        "  5. Keep each query concise (one sentence).\n"
        "  6. Output ONLY the queries. No numbering, no markdown, no preamble.\n\n"
        "Example:\n"
        '  Input:  "What is RAG?"\n'
        "  Output:\n"
        "  Explain Retrieval-Augmented Generation\n"
        "  How does Retrieval-Augmented Generation work?\n"
        "  RAG architecture components and pipeline\n"
        "  Document retrieval and generation in RAG systems"
    )

    # -- LLM ------------------------------------------------------------
    llm_provider: str = "openai"
    llm_model_name: str = "gpt-4o"
    llm_temperature: float = 0.0
    llm_max_tokens: int = 1024
    llm_timeout: int = 60
    llm_max_retries: int = 3

    claude_api_key: str = ""
    openai_api_key: str = ""
    gemini_api_key: str = ""

    # -- storage paths --------------------------------------------------
    index_storage_path: str = "storage/index.faiss"
    database_storage_path: str = "storage/metadata.db"

    # -- upload ---------------------------------------------------------
    upload_directory: str = "data/uploads"
    max_upload_size: int = 52_428_800  # 50 MiB
    supported_upload_extensions: str = ".pdf,.docx,.txt,.md"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
    )


settings = Settings()
