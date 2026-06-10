from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # Service runtime
    ENV: str = "production"
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000

    # LLM
    OPENAI_API_KEY: str
    LLM_MODEL: str = "gpt-4o-mini"
    LLM_MAX_TOKENS: int = 1000
    LLM_TEMPERATURE: float = 0.0

    # Embeddings
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    EMBEDDING_DIMENSIONS: int = 1536

    # Qdrant
    QDRANT_URL: str
    QDRANT_API_KEY: str | None = None
    QDRANT_COLLECTION: str = "documents"

    # Retrieval
    RETRIEVAL_TOP_K: int = 5
    RETRIEVAL_CANDIDATE_K: int = 20
    HYBRID_ALPHA: float = 0.5
    MAX_CONTEXT_TOKENS: int = 6000

    # Reranker sidecar
    RERANKER_URL: str = "http://reranker:8001"
    RERANKER_ENABLED: bool = True

    # Ingestion service
    INGESTION_URL: str = "http://ingestion:8002"

    # Semantic cache
    REDIS_URL: str = "redis://redis:6379"
    CACHE_SIMILARITY_THRESHOLD: float = 0.95
    CACHE_TTL_SECONDS: int = 3600

    # Tenancy
    DEFAULT_TENANT_ID: str = "default"

    # Observability
    LOG_LEVEL: str = "INFO"
    ENABLE_TRACING: bool = True


# BaseSettings fills required values from environment variables at runtime.
settings = Settings()  # pyright: ignore[reportCallIssue]
