from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, cast

import numpy as np
from openai import AsyncOpenAI
from redis.asyncio import Redis

from services.api.config import settings

logger = logging.getLogger(__name__)

MAX_CACHE_KEYS_PER_TENANT = 500


@dataclass(frozen=True)
class CacheHit:
    response: str
    query_text: str
    chunk_ids: list[str]
    context: str
    similarity_score: float
    created_at: str


class SemanticCache:
    def __init__(
        self,
        redis_client: Redis | None = None,
        embedding_client: AsyncOpenAI | None = None,
    ) -> None:
        self.redis = redis_client or Redis.from_url(settings.REDIS_URL, decode_responses=True)
        self.embedding_client = embedding_client or AsyncOpenAI(api_key=settings.OPENAI_API_KEY)

    async def get(self, query: str, tenant_id: str) -> CacheHit | None:
        query_embedding = await self._embed_query(query)
        keys = await self._tenant_keys(tenant_id)
        if not keys:
            return None

        rows = await self._load_hashes(keys)
        best_hit: CacheHit | None = None

        for row in rows:
            cached_hit = self._cache_hit_from_row(row, query_embedding)
            if cached_hit is None:
                continue

            if best_hit is None or cached_hit.similarity_score > best_hit.similarity_score:
                best_hit = cached_hit

        if (
            best_hit is not None
            and best_hit.similarity_score >= settings.CACHE_SIMILARITY_THRESHOLD
        ):
            return best_hit

        return None

    async def set(
        self,
        query: str,
        response: str,
        chunk_ids: list[str],
        tenant_id: str,
        context: str = "",
    ) -> None:
        query_embedding = await self._embed_query(query)
        key = self._cache_key(query, tenant_id)
        await self.redis.hset(
            key,
            mapping={
                "query_embedding": json.dumps(query_embedding),
                "response": response,
                "query_text": query,
                "retrieved_chunk_ids": json.dumps(chunk_ids),
                "context": context,
                "created_at": datetime.now(UTC).isoformat(),
            },
        )
        await self.redis.expire(key, settings.CACHE_TTL_SECONDS)

    async def invalidate_tenant(self, tenant_id: str) -> int:
        deleted_count = 0
        batch: list[str] = []

        async for key in self.redis.scan_iter(match=self._tenant_key_pattern(tenant_id)):
            batch.append(self._to_text(key))
            if len(batch) >= MAX_CACHE_KEYS_PER_TENANT:
                deleted_count += await self.redis.delete(*batch)
                batch.clear()

        if batch:
            deleted_count += await self.redis.delete(*batch)

        return deleted_count

    async def _tenant_keys(self, tenant_id: str) -> list[str]:
        keys: list[str] = []
        async for key in self.redis.scan_iter(
            match=self._tenant_key_pattern(tenant_id),
            count=MAX_CACHE_KEYS_PER_TENANT,
        ):
            keys.append(self._to_text(key))
            if len(keys) >= MAX_CACHE_KEYS_PER_TENANT:
                break
        return keys

    async def _load_hashes(self, keys: list[str]) -> list[dict[str, str]]:
        pipeline = self.redis.pipeline(transaction=False)
        for key in keys:
            pipeline.hgetall(key)

        rows = await pipeline.execute()
        return [self._string_dict(row) for row in rows if isinstance(row, dict)]

    async def _embed_query(self, query: str) -> list[float]:
        response = await self.embedding_client.embeddings.create(
            model=settings.EMBEDDING_MODEL,
            input=query,
        )
        return list(response.data[0].embedding)

    def _cache_hit_from_row(
        self,
        row: dict[str, str],
        query_embedding: list[float],
    ) -> CacheHit | None:
        try:
            cached_embedding = self._embedding_from_row(row)
            similarity_score = self._cosine_similarity(query_embedding, cached_embedding)
            return CacheHit(
                response=row["response"],
                query_text=row["query_text"],
                chunk_ids=self._chunk_ids_from_row(row),
                context=row.get("context", ""),
                similarity_score=similarity_score,
                created_at=row["created_at"],
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            logger.debug("skipping malformed semantic cache entry", exc_info=exc)
            return None

    def _embedding_from_row(self, row: dict[str, str]) -> list[float]:
        raw_embedding = json.loads(row["query_embedding"])
        if not isinstance(raw_embedding, list):
            raise TypeError("cached embedding must be a list")
        return [float(value) for value in raw_embedding]

    def _chunk_ids_from_row(self, row: dict[str, str]) -> list[str]:
        raw_chunk_ids = json.loads(row.get("retrieved_chunk_ids", "[]"))
        if not isinstance(raw_chunk_ids, list):
            raise TypeError("cached chunk ids must be a list")
        return [str(chunk_id) for chunk_id in raw_chunk_ids]

    def _cosine_similarity(self, left: list[float], right: list[float]) -> float:
        if len(left) != len(right):
            return 0.0

        left_vector = np.asarray(left, dtype=np.float32)
        right_vector = np.asarray(right, dtype=np.float32)
        denominator = float(np.linalg.norm(left_vector) * np.linalg.norm(right_vector))
        if denominator == 0.0:
            return 0.0

        # Base Redis has no vector index. For larger tenants, replace this scan with
        # Redis Stack vector search or move cache vectors into pgvector.
        return float(np.dot(left_vector, right_vector) / denominator)

    def _cache_key(self, query: str, tenant_id: str) -> str:
        query_hash = hashlib.sha256(query.encode("utf-8")).hexdigest()
        return f"cache:{tenant_id}:{query_hash}"

    def _tenant_key_pattern(self, tenant_id: str) -> str:
        return f"cache:{tenant_id}:*"

    def _string_dict(self, row: dict[Any, Any]) -> dict[str, str]:
        return {self._to_text(key): self._to_text(value) for key, value in row.items()}

    def _to_text(self, value: Any) -> str:
        if isinstance(value, bytes):
            return value.decode("utf-8")
        return cast(str, value)
