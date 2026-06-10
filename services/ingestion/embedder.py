from __future__ import annotations

import json
import logging

from openai import AsyncOpenAI, RateLimitError
from services.ingestion.config import settings
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)


class BatchEmbedder:
    def __init__(self, client: AsyncOpenAI | None = None, model: str | None = None) -> None:
        self.client = client or AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        self.model = model or settings.EMBEDDING_MODEL

    async def embed_batch(self, texts: list[str], batch_size: int = 100) -> list[list[float]]:
        embeddings: list[list[float]] = []

        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            embeddings.extend(await self._embed_with_retry(batch))

        return embeddings

    @retry(
        retry=retry_if_exception_type(RateLimitError),
        wait=wait_exponential(multiplier=1, min=1, max=4),
        stop=stop_after_attempt(3),
        reraise=True,
    )
    async def _embed_with_retry(self, texts: list[str]) -> list[list[float]]:
        response = await self.client.embeddings.create(model=self.model, input=texts)

        total_tokens = response.usage.total_tokens if response.usage is not None else 0
        logger.info(
            json.dumps(
                {
                    "event": "openai_embedding_batch",
                    "batch_size": len(texts),
                    "total_tokens": total_tokens,
                }
            )
        )

        return [item.embedding for item in response.data]
