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
        "  Output: \"Explain Retrieval-Augmented Generation (RAG), including its indexing, "
        'retrieval, reranking, and LLM generation components."\n\n'
        '  Input:  "What are the benefits?"\n'
        "  Output: \"What are the benefits of using a RAG system "
        'for enterprise knowledge management?"'
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
