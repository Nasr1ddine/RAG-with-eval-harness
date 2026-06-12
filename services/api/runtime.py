from __future__ import annotations

import asyncio
import threading
from collections.abc import Coroutine
from dataclasses import dataclass
from typing import Any, TypeVar

import httpx
import structlog
from openai import AsyncOpenAI

from services.api.cache import SemanticCache
from services.api.config import settings
from services.api.errors import ServiceUnavailableError
from services.api.metrics import record_rag_request
from services.api.observability import (
    bind_tenant_context,
    configure_logging,
    request_context_headers,
)
from services.api.pipeline import QueryRequest, QueryResponse, RAGPipeline
from services.api.retrieval import HybridRetriever, ParentExpander

configure_logging(service_name="api", log_level=settings.LOG_LEVEL, environment=settings.ENV)
logger = structlog.get_logger(__name__)
T = TypeVar("T")


@dataclass
class RuntimeResources:
    pipeline: RAGPipeline
    http_client: httpx.AsyncClient
    cache: SemanticCache
    llm_client: AsyncOpenAI


_runtime: RuntimeResources | None = None
_sync_runner: AsyncRuntimeRunner | None = None


async def initialize_runtime() -> RuntimeResources:
    global _runtime
    if _runtime is not None:
        return _runtime

    llm_client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
    http_client = httpx.AsyncClient()
    cache = SemanticCache(embedding_client=llm_client)
    pipeline = RAGPipeline(
        retriever=HybridRetriever(),
        expander=ParentExpander(tenant_id=settings.DEFAULT_TENANT_ID),
        cache=cache,
        reranker_client=http_client,
        llm_client=llm_client,
    )

    await _warm_up_reranker(http_client)

    _runtime = RuntimeResources(
        pipeline=pipeline,
        http_client=http_client,
        cache=cache,
        llm_client=llm_client,
    )
    return _runtime


async def shutdown_runtime() -> None:
    global _runtime
    if _runtime is None:
        return

    await _runtime.http_client.aclose()
    await _runtime.cache.redis.aclose()
    await _runtime.llm_client.close()
    _runtime = None


def get_runtime() -> RuntimeResources:
    if _runtime is None:
        raise RuntimeError("RAG runtime is not initialized")
    return _runtime


async def run_query(request: QueryRequest) -> QueryResponse:
    bind_tenant_context(request.tenant_id)
    runtime = get_runtime()
    response = await runtime.pipeline.query(request)
    record_rag_request(
        tenant_id=request.tenant_id,
        cache_hit=response.cache_hit,
        latency_ms=response.latency_ms,
        retrieval_candidates=response.retrieval_count,
        context_tokens=response.context_tokens,
    )
    logger.info(
        "rag query completed",
        tenant_id=request.tenant_id,
        cache_hit=response.cache_hit,
        latency_ms=response.latency_ms,
        token_count=response.context_tokens,
    )
    return response


async def ingest_document(
    *,
    filename: str,
    content: bytes,
    content_type: str,
    tenant_id: str,
) -> dict[str, Any]:
    bind_tenant_context(tenant_id)
    runtime = get_runtime()
    files = {
        "file": (
            filename,
            content,
            content_type or "application/octet-stream",
        )
    }
    data = {"tenant_id": tenant_id}

    try:
        response = await runtime.http_client.post(
            f"{settings.INGESTION_URL.rstrip('/')}/ingest",
            files=files,
            data=data,
            headers=request_context_headers(),
        )
    except httpx.HTTPError as exc:
        raise ServiceUnavailableError("Ingestion service unavailable") from exc

    if response.is_error:
        raise ServiceUnavailableError(str(_error_detail(response)))

    payload = response.json()
    if not isinstance(payload, dict):
        raise ServiceUnavailableError("Ingestion service returned an invalid response")
    return payload


def run_query_sync(request: QueryRequest) -> QueryResponse:
    return get_sync_runtime_runner().query(request)


def ingest_document_sync(
    *,
    filename: str,
    content: bytes,
    content_type: str,
    tenant_id: str,
) -> dict[str, Any]:
    return get_sync_runtime_runner().ingest_document(
        filename=filename,
        content=content,
        content_type=content_type,
        tenant_id=tenant_id,
    )


class AsyncRuntimeRunner:
    """Synchronous bridge that keeps async clients on one event loop."""

    def __init__(self) -> None:
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._run_loop,
            name="rag-api-runtime",
            daemon=True,
        )
        self._closed = False
        self._thread.start()
        self.runtime = self._run(initialize_runtime())

    def query(self, request: QueryRequest) -> QueryResponse:
        return self._run(run_query(request))

    def ingest_document(
        self,
        *,
        filename: str,
        content: bytes,
        content_type: str,
        tenant_id: str,
    ) -> dict[str, Any]:
        return self._run(
            ingest_document(
                filename=filename,
                content=content,
                content_type=content_type,
                tenant_id=tenant_id,
            )
        )

    def close(self) -> None:
        if self._closed:
            return

        try:
            self._run(shutdown_runtime())
        finally:
            self._closed = True
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._thread.join(timeout=5.0)
            self._loop.close()

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def _run(self, coroutine: Coroutine[Any, Any, T]) -> T:
        if self._closed:
            raise RuntimeError("RAG runtime runner is closed")

        future = asyncio.run_coroutine_threadsafe(coroutine, self._loop)
        return future.result()


def get_sync_runtime_runner() -> AsyncRuntimeRunner:
    global _sync_runner
    if _sync_runner is None or _sync_runner._closed:
        _sync_runner = AsyncRuntimeRunner()
    return _sync_runner


def close_sync_runtime_runner() -> None:
    global _sync_runner
    if _sync_runner is None:
        return

    _sync_runner.close()
    _sync_runner = None


async def _run_query_with_runtime(request: QueryRequest) -> QueryResponse:
    await initialize_runtime()
    return await run_query(request)


async def _ingest_document_with_runtime(
    *,
    filename: str,
    content: bytes,
    content_type: str,
    tenant_id: str,
) -> dict[str, Any]:
    await initialize_runtime()
    return await ingest_document(
        filename=filename,
        content=content,
        content_type=content_type,
        tenant_id=tenant_id,
    )


async def _warm_up_reranker(http_client: httpx.AsyncClient) -> None:
    if not settings.RERANKER_ENABLED:
        return

    try:
        response = await http_client.post(
            f"{settings.RERANKER_URL.rstrip('/')}/health",
            timeout=5.0,
        )
        response.raise_for_status()
    except httpx.HTTPError as exc:
        logger.warning("reranker warmup failed", error=str(exc))


def _error_detail(response: httpx.Response) -> Any:
    try:
        payload = response.json()
    except ValueError:
        return response.text
    if isinstance(payload, dict) and "detail" in payload:
        return payload["detail"]
    return payload
