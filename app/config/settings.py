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

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
    )


settings = Settings()
