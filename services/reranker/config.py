from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # Service runtime
    ENV: str = "production"
    RERANKER_HOST: str = "0.0.0.0"
    RERANKER_PORT: int = 8001

    # Cross-encoder model
    RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # Observability
    LOG_LEVEL: str = "INFO"
    ENABLE_TRACING: bool = True


settings = Settings()
