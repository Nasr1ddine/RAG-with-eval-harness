from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, Field

from services.reranker.config import settings
from services.reranker.model import load_model, model_name, score_candidates
from services.reranker.observability import RequestLoggingMiddleware, configure_logging

_rerank_lock = asyncio.Lock()
configure_logging(service_name="reranker", log_level=settings.LOG_LEVEL, environment=settings.ENV)


class Candidate(BaseModel):
    id: str
    text: str


class RerankRequest(BaseModel):
    query: str
    candidates: list[Candidate] = Field(max_length=50)


class RankedCandidate(BaseModel):
    id: str
    score: float


class RerankResponse(BaseModel):
    ranked: list[RankedCandidate]


class HealthResponse(BaseModel):
    status: str
    model: str


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    load_model()
    yield


app = FastAPI(title="Reranker Service", lifespan=lifespan)
app.add_middleware(
    RequestLoggingMiddleware,
    service_name="reranker",
    default_tenant_id="unknown",
)


@app.post("/rerank", response_model=RerankResponse)
async def rerank(request: RerankRequest) -> RerankResponse:
    async with _rerank_lock:
        scores = score_candidates(
            request.query,
            [candidate.text for candidate in request.candidates],
        )

    ranked = sorted(
        (
            RankedCandidate(id=candidate.id, score=score)
            for candidate, score in zip(request.candidates, scores, strict=True)
        ),
        key=lambda candidate: candidate.score,
        reverse=True,
    )
    return RerankResponse(ranked=ranked)


@app.post("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(status="ok", model=model_name())


def main() -> None:
    uvicorn.run(
        "services.reranker.main:app",
        host=settings.RERANKER_HOST,
        port=settings.RERANKER_PORT,
    )


if __name__ == "__main__":
    main()
