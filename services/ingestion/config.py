from typing import Self

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # Service runtime
    ENV: str = "production"
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8002
    PORT: int | None = None
    # Embeddings
    OPENAI_API_KEY: str
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    EMBEDDING_DIMENSIONS: int = 1536

    # Qdrant
    QDRANT_URL: str
    QDRANT_API_KEY: str | None = None
    QDRANT_COLLECTION: str = "documents"

    # Document chunking
    INGESTION_PARENT_CHUNK_SIZE: int = 1500
    INGESTION_CHILD_CHUNK_SIZE: int = 300
    INGESTION_CHUNK_OVERLAP: int = 50
    INGESTION_BATCH_SIZE: int = 100

    # Tenancy
    DEFAULT_TENANT_ID: str = "default"

    # Observability
    LOG_LEVEL: str = "INFO"
    ENABLE_TRACING: bool = True

    @model_validator(mode="after")
    def resolve_listening_port(self) -> Self:
        if self.PORT is not None:
            self.API_PORT = self.PORT
        return self


settings = Settings()
