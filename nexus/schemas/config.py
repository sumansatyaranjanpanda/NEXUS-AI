"""Application settings via pydantic-settings. All secrets come from env vars."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central configuration — mirrors .env.example exactly."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- LLM (OpenRouter, OpenAI-compatible gateway) ---
    openrouter_api_key: str
    openrouter_model: str = "anthropic/claude-sonnet-4.6"
    openrouter_base_url: str = "https://openrouter.ai/api/v1"

    # --- Qdrant ---
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "nexus_kb"

    # --- Redis ---
    redis_url: str = "redis://localhost:6379/0"

    # --- Cohere ---
    cohere_api_key: str = ""

    # --- RAG ---
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dim: int = 384
    sparse_model: str = "prithivida/Splade_PP_en_v1"
    chunk_size: int = 512
    chunk_overlap: int = 50
    retrieval_top_k: int = 10
    rerank_top_k: int = 5

    # --- Web Search (Tavily) ---
    tavily_api_key: str = ""  # Set to enable real web search in research agent

    # --- Evaluation & Tracing ---
    langfuse_secret_key: str = ""
    langfuse_public_key: str = ""
    langfuse_base_url: str = "https://cloud.langfuse.com"

    # --- App ---
    log_level: str = "INFO"
    app_env: str = "development"
