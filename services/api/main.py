from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import httpx
import structlog
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from openai import AsyncOpenAI
from prometheus_client import CONTENT_TYPE_LATEST

from services.api.cache import SemanticCache
from services.api.config import settings
from services.api.metrics import record_rag_request, render_metrics
from services.api.observability import (
    RequestLoggingMiddleware,
    bind_tenant_context,
    configure_logging,
    request_context_headers,
)
from services.api.pipeline import QueryRequest, QueryResponse, RAGPipeline
from services.api.retrieval import HybridRetriever, ParentExpander

configure_logging(service_name="api", log_level=settings.LOG_LEVEL, environment=settings.ENV)
logger = structlog.get_logger(__name__)

STATIC_DIR = Path(__file__).parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
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

    app.state.pipeline = pipeline
    app.state.http_client = http_client
    app.state.cache = cache
    app.state.llm_client = llm_client

    await _warm_up_reranker(http_client)

    try:
        yield
    finally:
        await http_client.aclose()
        await cache.redis.aclose()
        await llm_client.close()


app = FastAPI(title="RAG API Service", lifespan=lifespan)
app.add_middleware(
    RequestLoggingMiddleware,
    service_name="api",
    default_tenant_id=settings.DEFAULT_TENANT_ID,
)

if STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.post("/query", response_model=QueryResponse)
async def query(request_body: QueryRequest, request: Request) -> QueryResponse:
    bind_tenant_context(request_body.tenant_id)
    request.state.tenant_id = request_body.tenant_id
    pipeline = _pipeline(request)
    response = await pipeline.query(request_body)
    request.state.rag_metrics = {
        "tenant_id": request_body.tenant_id,
        "cache_hit": response.cache_hit,
        "token_count": response.context_tokens,
    }
    record_rag_request(
        tenant_id=request_body.tenant_id,
        cache_hit=response.cache_hit,
        latency_ms=response.latency_ms,
        retrieval_candidates=response.retrieval_count,
        context_tokens=response.context_tokens,
    )
    return response


@app.post("/ingest")
async def ingest(
    request: Request,
    file: UploadFile = File(...),
    tenant_id: str = Form(...),
) -> JSONResponse:
    bind_tenant_context(tenant_id)
    request.state.tenant_id = tenant_id
    request.state.rag_metrics = {
        "tenant_id": tenant_id,
        "cache_hit": False,
        "token_count": 0,
    }
    content = await file.read()
    files = {
        "file": (
            file.filename or "upload",
            content,
            file.content_type or "application/octet-stream",
        )
    }
    data = {"tenant_id": tenant_id}

    try:
        response = await _http_client(request).post(
            f"{settings.INGESTION_URL.rstrip('/')}/ingest",
            files=files,
            data=data,
            headers=request_context_headers(),
        )
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=503, detail="Ingestion service unavailable") from exc

    if response.is_error:
        raise HTTPException(status_code=response.status_code, detail=_error_detail(response))

    return JSONResponse(status_code=response.status_code, content=response.json())


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/metrics")
async def metrics() -> Response:
    return Response(content=render_metrics(), media_type=CONTENT_TYPE_LATEST)


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


def _pipeline(request: Request) -> RAGPipeline:
    pipeline = request.app.state.pipeline
    if not isinstance(pipeline, RAGPipeline):
        raise RuntimeError("RAG pipeline is not initialized")
    return pipeline


def _http_client(request: Request) -> httpx.AsyncClient:
    http_client = request.app.state.http_client
    if not isinstance(http_client, httpx.AsyncClient):
        raise RuntimeError("HTTP client is not initialized")
    return http_client


def _error_detail(response: httpx.Response) -> Any:
    try:
        payload = response.json()
    except ValueError:
        return response.text
    if isinstance(payload, dict) and "detail" in payload:
        return payload["detail"]
    return payload


def main() -> None:
    uvicorn.run(
        "services.api.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
    )


if __name__ == "__main__":
    main()
