from __future__ import annotations

import asyncio
import time
from typing import Any

import httpx
import structlog
import tiktoken
from openai import AsyncOpenAI
from pydantic import BaseModel, Field

from services.api.cache import SemanticCache
from services.api.config import settings
from services.api.errors import ServiceUnavailableError
from services.api.observability import request_context_headers
from services.api.retrieval import ExpandedChunk, HybridRetriever, ParentExpander, RetrievedChunk

logger = structlog.get_logger(__name__)

QUERY_REWRITE_SYSTEM_PROMPT = (
    "Rewrite the user query for document retrieval. "
    "Return only the rewritten query, no explanation."
)
ANSWER_SYSTEM_PROMPT = (
    "Answer only from the provided context. Cite sources by referencing [doc_id]. "
    "If the context does not contain the answer, say: "
    "'I don't have information on this in the provided documents.' Do not speculate."
)
RERANKER_TIMEOUT_SECONDS = 5.0


class QueryRequest(BaseModel):
    query: str = Field(min_length=1)
    tenant_id: str = Field(min_length=1)
    top_k: int | None = Field(default=None, ge=1, le=50)


class QuerySource(BaseModel):
    chunk_id: str
    score: float
    text: str = ""


class QueryResponse(BaseModel):
    answer: str
    sources: list[QuerySource]
    context: str = ""
    cache_hit: bool
    latency_ms: int
    retrieval_count: int
    reranked_count: int
    context_tokens: int


class RAGPipeline:
    def __init__(
        self,
        retriever: HybridRetriever,
        expander: ParentExpander,
        cache: SemanticCache,
        reranker_client: httpx.AsyncClient,
        llm_client: AsyncOpenAI,
    ) -> None:
        self.retriever = retriever
        self.expander = expander
        self.cache = cache
        self.reranker_client = reranker_client
        self.llm_client = llm_client
        self.encoding = self._encoding_for_model(settings.LLM_MODEL)

    async def query(self, request: QueryRequest) -> QueryResponse:
        start_time = time.perf_counter()

        cache_hit = await self.cache.get(request.query, request.tenant_id)
        if cache_hit is not None:
            latency_ms = self._latency_ms(start_time)
            logger.info(
                "rag query served from semantic cache",
                tenant_id=request.tenant_id,
                cache_hit=True,
            )
            return QueryResponse(
                answer=cache_hit.response,
                sources=[
                    QuerySource(chunk_id=chunk_id, score=cache_hit.similarity_score)
                    for chunk_id in cache_hit.chunk_ids
                ],
                context=cache_hit.context,
                cache_hit=True,
                latency_ms=latency_ms,
                retrieval_count=0,
                reranked_count=0,
                context_tokens=0,
            )

        top_k = request.top_k or settings.RETRIEVAL_TOP_K
        rewritten_query = await self._rewrite_query(request.query)
        retrieved = await asyncio.to_thread(
            self.retriever.retrieve,
            rewritten_query,
            request.tenant_id,
            settings.RETRIEVAL_CANDIDATE_K,
        )
        reranked = await self._rerank(rewritten_query, retrieved, top_k)
        expanded = await asyncio.to_thread(self.expander.expand, reranked)
        context, context_tokens, context_chunks = self._assemble_context(expanded)
        answer = await self._generate_answer(request.query, context)
        sources = [
            QuerySource(chunk_id=chunk.chunk_id, score=chunk.score, text=chunk.parent_text)
            for chunk in context_chunks
        ]

        await self.cache.set(
            query=request.query,
            response=answer,
            chunk_ids=[source.chunk_id for source in sources],
            tenant_id=request.tenant_id,
            context=context,
        )

        return QueryResponse(
            answer=answer,
            sources=sources,
            context=context,
            cache_hit=False,
            latency_ms=self._latency_ms(start_time),
            retrieval_count=len(retrieved),
            reranked_count=len(reranked),
            context_tokens=context_tokens,
        )

    async def _rewrite_query(self, query: str) -> str:
        response = await self.llm_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": QUERY_REWRITE_SYSTEM_PROMPT},
                {"role": "user", "content": query},
            ],
            temperature=0.0,
            max_tokens=128,
        )
        rewritten_query = response.choices[0].message.content
        if rewritten_query is None:
            return query
        return rewritten_query.strip() or query

    async def _rerank(
        self,
        query: str,
        candidates: list[RetrievedChunk],
        top_k: int,
    ) -> list[RetrievedChunk]:
        if not settings.RERANKER_ENABLED or not candidates:
            return candidates[:top_k]

        payload = {
            "query": query,
            "candidates": [
                {"id": candidate.chunk_id, "text": candidate.text} for candidate in candidates
            ],
        }

        try:
            response = await self.reranker_client.post(
                f"{settings.RERANKER_URL.rstrip('/')}/rerank",
                json=payload,
                headers=request_context_headers(),
                timeout=RERANKER_TIMEOUT_SECONDS,
            )
            response.raise_for_status()
            response_payload: Any = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise ServiceUnavailableError("Reranker service unavailable") from exc

        ranked_items = (
            response_payload.get("ranked") if isinstance(response_payload, dict) else None
        )
        if not isinstance(ranked_items, list):
            raise ServiceUnavailableError("Reranker service unavailable")

        candidates_by_id = {candidate.chunk_id: candidate for candidate in candidates}
        ranked: list[RetrievedChunk] = []
        seen_ids: set[str] = set()
        try:
            for item in ranked_items:
                if not isinstance(item, dict):
                    raise TypeError("ranked item must be an object")
                chunk_id = str(item["id"])
                candidate = candidates_by_id.get(chunk_id)
                if candidate is None:
                    continue
                seen_ids.add(chunk_id)
                ranked.append(
                    RetrievedChunk(
                        chunk_id=candidate.chunk_id,
                        text=candidate.text,
                        score=float(item["score"]),
                        metadata=candidate.metadata,
                        parent_id=candidate.parent_id,
                    )
                )
        except (KeyError, TypeError, ValueError) as exc:
            raise ServiceUnavailableError("Reranker service unavailable") from exc

        ranked.extend(candidate for candidate in candidates if candidate.chunk_id not in seen_ids)
        return ranked[:top_k]

    def _assemble_context(
        self,
        chunks: list[ExpandedChunk],
    ) -> tuple[str, int, list[ExpandedChunk]]:
        selected = list(chunks)
        selected.sort(key=lambda chunk: chunk.score, reverse=True)
        context = self._format_context(selected)
        token_count = self._token_count(context)

        while len(selected) > 1 and token_count > settings.MAX_CONTEXT_TOKENS:
            selected.pop()
            context = self._format_context(selected)
            token_count = self._token_count(context)

        if token_count <= settings.MAX_CONTEXT_TOKENS:
            return context, token_count, selected

        truncated_context = self.encoding.decode(
            self.encoding.encode(context)[: settings.MAX_CONTEXT_TOKENS]
        )
        return truncated_context, self._token_count(truncated_context), selected

    def _format_context(self, chunks: list[ExpandedChunk]) -> str:
        return "\n\n".join(f"[{chunk.chunk_id}]\n{chunk.parent_text.strip()}" for chunk in chunks)

    async def _generate_answer(self, query: str, context: str) -> str:
        response = await self.llm_client.chat.completions.create(
            model=settings.LLM_MODEL,
            messages=[
                {"role": "system", "content": ANSWER_SYSTEM_PROMPT},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion:\n{query}"},
            ],
            temperature=settings.LLM_TEMPERATURE,
            max_tokens=settings.LLM_MAX_TOKENS,
        )
        answer = response.choices[0].message.content
        return answer.strip() if answer is not None else ""

    def _token_count(self, text: str) -> int:
        return len(self.encoding.encode(text))

    def _encoding_for_model(self, model: str) -> tiktoken.Encoding:
        try:
            return tiktoken.encoding_for_model(model)
        except KeyError:
            return tiktoken.get_encoding("cl100k_base")

    def _latency_ms(self, start_time: float) -> int:
        return round((time.perf_counter() - start_time) * 1000)
