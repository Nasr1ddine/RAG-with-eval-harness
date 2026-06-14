from __future__ import annotations

import os
from typing import Any

import pytest

os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("QDRANT_URL", "http://qdrant:6333")
os.environ.setdefault("REDIS_URL", "redis://redis:6379")

from services.api.cache import SemanticCache


class FakeRedis:
    def __init__(self, keys: list[str]) -> None:
        self.keys = keys
        self.deleted: list[str] = []

    async def scan_iter(self, *_args: Any, **_kwargs: Any) -> Any:
        for key in self.keys:
            yield key

    async def delete(self, *keys: str) -> int:
        self.deleted.extend(keys)
        return len(keys)


@pytest.mark.asyncio
async def test_invalidate_tenant_deletes_matching_cache_keys() -> None:
    redis = FakeRedis(["cache:tenant-a:one", "cache:tenant-a:two"])
    cache = SemanticCache(redis_client=redis)  # type: ignore[arg-type]

    deleted_count = await cache.invalidate_tenant("tenant-a")

    assert deleted_count == 2
    assert redis.deleted == ["cache:tenant-a:one", "cache:tenant-a:two"]
