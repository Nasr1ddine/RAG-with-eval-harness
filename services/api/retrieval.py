from __future__ import annotations

import logging
import time
from dataclasses import dataclass, replace
from typing import Any

import bm25s  # type: ignore[import-not-found,import-untyped]
import httpx
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.http import models

from services.api.config import settings
from services.api.errors import ServiceUnavailableError
from services.ingestion.chunker import Chunk

logger = logging.getLogger(__name__)

RRF_K = 60
EMBEDDING_METADATA_KEY = "_embedding"
RERANKER_MAX_CANDIDATES = 50
RERANKER_TIMEOUT_SECONDS = 5.0


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: str
    text: str
    score: float
    metadata: dict[str, Any]
    parent_id: str


@dataclass(frozen=True)
class ExpandedChunk(RetrievedChunk):
    parent_text: str


async def rerank_chunks(query: str, candidates: list[RetrievedChunk]) -> list[RetrievedChunk]:
    if not settings.RERANKER_ENABLED or not candidates:
        return candidates

    limited_candidates = candidates[:RERANKER_MAX_CANDIDATES]
    payload = {
        "query": query,
        "candidates": [
            {"id": candidate.chunk_id, "text": candidate.text} for candidate in limited_candidates
        ],
    }

    try:
        async with httpx.AsyncClient(timeout=RERANKER_TIMEOUT_SECONDS) as client:
            response = await client.post(
                f"{settings.RERANKER_URL.rstrip('/')}/rerank",
                json=payload,
            )
            response.raise_for_status()
            response_payload: Any = response.json()
    except (httpx.HTTPError, ValueError) as exc:
        raise ServiceUnavailableError("Reranker service unavailable") from exc

    return _ranked_chunks_from_response(response_payload, limited_candidates)


def _ranked_chunks_from_response(
    response_payload: Any,
    candidates: list[RetrievedChunk],
) -> list[RetrievedChunk]:
    ranked_items = response_payload.get("ranked") if isinstance(response_payload, dict) else None
    if not isinstance(ranked_items, list):
        raise ServiceUnavailableError("Reranker service unavailable")

    candidates_by_id = {candidate.chunk_id: candidate for candidate in candidates}
    ranked_chunks: list[RetrievedChunk] = []
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
            ranked_chunks.append(replace(candidate, score=float(item["score"])))
    except (KeyError, TypeError, ValueError) as exc:
        raise ServiceUnavailableError("Reranker service unavailable") from exc

    ranked_chunks.extend(
        candidate for candidate in candidates if candidate.chunk_id not in seen_ids
    )
    return ranked_chunks


class HybridRetriever:
    def __init__(
        self,
        client: QdrantClient | None = None,
        embedding_client: OpenAI | None = None,
        collection_name: str | None = None,
        alpha: float | None = None,
        candidate_k: int | None = None,
        chunks: list[Chunk] | None = None,
    ) -> None:
        self.client = client or QdrantClient(
            url=settings.QDRANT_URL,
            api_key=settings.QDRANT_API_KEY,
        )
        base_collection_name = collection_name or settings.QDRANT_COLLECTION
        self.children_collection_name = f"{base_collection_name}_children"
        self.embedding_client = embedding_client or OpenAI(api_key=settings.OPENAI_API_KEY)
        self.alpha = settings.HYBRID_ALPHA if alpha is None else alpha
        self.candidate_k = settings.RETRIEVAL_CANDIDATE_K if candidate_k is None else candidate_k
        self._bm25: Any | None = None
        self._bm25_chunks: list[Chunk] = []
        self._warned_empty_bm25 = False

        if chunks is not None:
            self.rebuild_index(chunks)

    @property
    def bm25_corpus_size(self) -> int:
        return len(self._bm25_chunks)

    def rebuild_index(self, chunks: list[Chunk]) -> None:
        child_chunks = [chunk for chunk in chunks if chunk.metadata.get("chunk_type") == "child"]
        if not child_chunks and chunks:
            child_chunks = chunks

        self._bm25_chunks = child_chunks
        self._warned_empty_bm25 = False
        if not child_chunks:
            self._bm25 = None
            return

        tokenized_corpus = bm25s.tokenize([chunk.text for chunk in child_chunks])
        self._bm25 = bm25s.BM25()
        self._bm25.index(tokenized_corpus)

    def rebuild_index_from_qdrant(self) -> int:
        """Refresh BM25 from persisted child chunks after startup or ingestion."""
        start_time = time.perf_counter()
        chunks = self._load_child_chunks_from_qdrant()
        self.rebuild_index(chunks)

        log_context = {
            "collection_name": self.children_collection_name,
            "corpus_size": self.bm25_corpus_size,
            "latency_seconds": round(time.perf_counter() - start_time, 4),
        }
        if self.bm25_corpus_size == 0:
            logger.warning(
                "bm25 index is empty; hybrid retrieval will run dense-only",
                extra=log_context,
            )
        else:
            logger.info("bm25 index rebuilt", extra=log_context)
        return self.bm25_corpus_size

    def retrieve(self, query: str, tenant_id: str, top_k: int) -> list[RetrievedChunk]:
        start_time = time.perf_counter()
        if top_k <= 0:
            logger.debug(
                "hybrid retrieval completed",
                extra={
                    "tenant_id": tenant_id,
                    "latency_seconds": round(time.perf_counter() - start_time, 4),
                    "dense_candidate_count": 0,
                    "bm25_candidate_count": 0,
                    "candidate_count": 0,
                },
            )
            return []

        candidate_limit = max(top_k, self.candidate_k)

        dense_candidates = self._dense_search(query, tenant_id, candidate_limit)
        bm25_candidates = self._bm25_search(query, tenant_id, candidate_limit)
        fused_candidates = self._fuse_candidates(dense_candidates, bm25_candidates)
        results = fused_candidates[:top_k]

        logger.debug(
            "hybrid retrieval completed",
            extra={
                "tenant_id": tenant_id,
                "latency_seconds": round(time.perf_counter() - start_time, 4),
                "dense_candidate_count": len(dense_candidates),
                "bm25_candidate_count": len(bm25_candidates),
                "candidate_count": len(fused_candidates),
            },
        )
        return results

    async def retrieve_reranked(
        self,
        query: str,
        tenant_id: str,
        top_k: int,
    ) -> list[RetrievedChunk]:
        if top_k <= 0:
            return []

        candidate_limit = min(max(top_k, self.candidate_k), RERANKER_MAX_CANDIDATES)
        candidates = self.retrieve(query, tenant_id, candidate_limit)
        ranked_candidates = await rerank_chunks(query, candidates)
        return ranked_candidates[:top_k]

    def _dense_search(
        self,
        query: str,
        tenant_id: str,
        limit: int,
    ) -> list[RetrievedChunk]:
        query_vector = self._embed_query(query)
        result = self.client.query_points(
            collection_name=self.children_collection_name,
            query=query_vector,
            query_filter=self._tenant_filter(tenant_id),
            limit=limit,
            with_payload=True,
        )

        chunks: list[RetrievedChunk] = []
        for point in result.points:
            chunk = self._point_to_retrieved_chunk(point, tenant_id, score=0.0)
            if chunk is not None:
                chunks.append(chunk)
        return chunks

    def _bm25_search(self, query: str, tenant_id: str, limit: int) -> list[RetrievedChunk]:
        if self._bm25 is None or not self._bm25_chunks:
            if not self._warned_empty_bm25:
                logger.warning(
                    "bm25 index is empty during retrieval; returning dense-only candidates",
                    extra={"tenant_id": tenant_id},
                )
                self._warned_empty_bm25 = True
            return []

        query_tokens = bm25s.tokenize(query)
        result_rows, _score_rows = self._bm25.retrieve(
            query_tokens,
            corpus=list(range(len(self._bm25_chunks))),
            k=min(limit, len(self._bm25_chunks)),
        )
        result_indices = self._first_result_row(result_rows)

        chunks: list[RetrievedChunk] = []
        for result_index in result_indices:
            chunk = self._bm25_chunks[int(result_index)]
            if self._chunk_tenant_id(chunk) != tenant_id:
                continue

            parent_id = self._chunk_parent_id(chunk)
            if parent_id is None:
                continue

            metadata = self._chunk_metadata(chunk, tenant_id, parent_id)
            chunks.append(
                RetrievedChunk(
                    chunk_id=chunk.id,
                    text=chunk.text,
                    score=0.0,
                    metadata=metadata,
                    parent_id=str(parent_id),
                )
            )

        return chunks

    def _fuse_candidates(
        self,
        dense_candidates: list[RetrievedChunk],
        bm25_candidates: list[RetrievedChunk],
    ) -> list[RetrievedChunk]:
        chunk_by_id: dict[str, RetrievedChunk] = {}
        scores: dict[str, float] = {}

        for rank, chunk in enumerate(dense_candidates, start=1):
            chunk_by_id[chunk.chunk_id] = chunk
            scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0.0) + (
                self.alpha * self._rrf_score(rank)
            )

        for rank, chunk in enumerate(bm25_candidates, start=1):
            chunk_by_id.setdefault(chunk.chunk_id, chunk)
            scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0.0) + (
                (1.0 - self.alpha) * self._rrf_score(rank)
            )

        return sorted(
            (
                RetrievedChunk(
                    chunk_id=chunk.chunk_id,
                    text=chunk.text,
                    score=scores[chunk_id],
                    metadata=chunk.metadata,
                    parent_id=chunk.parent_id,
                )
                for chunk_id, chunk in chunk_by_id.items()
            ),
            key=lambda chunk: chunk.score,
            reverse=True,
        )

    def _embed_query(self, query: str) -> list[float]:
        response = self.embedding_client.embeddings.create(
            model=settings.EMBEDDING_MODEL,
            input=query,
        )
        return response.data[0].embedding

    def _load_child_chunks_from_qdrant(self) -> list[Chunk]:
        if not self.client.collection_exists(self.children_collection_name):
            return []

        chunks: list[Chunk] = []
        offset: Any | None = None
        while True:
            records, next_page = self.client.scroll(
                collection_name=self.children_collection_name,
                offset=offset,
                limit=256,
                with_payload=True,
                with_vectors=False,
            )
            for record in records:
                chunk = self._record_to_chunk(record)
                if chunk is not None:
                    chunks.append(chunk)

            if next_page is None:
                break
            offset = next_page

        return chunks

    def _record_to_chunk(self, record: Any) -> Chunk | None:
        payload = record.payload or {}
        text = payload.get("text")
        if not isinstance(text, str):
            return None

        tenant_id = self._payload_tenant_id(payload)
        parent_id = payload.get("parent_id")
        metadata = self._payload_metadata(
            payload,
            tenant_id=tenant_id or "",
            parent_id=str(parent_id) if parent_id is not None else None,
        )
        if tenant_id is None:
            metadata.pop("tenant_id", None)

        token_count = payload.get("token_count")
        if not isinstance(token_count, int):
            token_count = 0

        return Chunk(
            id=str(record.id),
            text=text,
            metadata=metadata,
            token_count=token_count,
        )

    def _point_to_retrieved_chunk(
        self,
        point: Any,
        tenant_id: str,
        score: float,
    ) -> RetrievedChunk | None:
        payload = point.payload or {}
        if self._payload_tenant_id(payload) != tenant_id:
            return None

        parent_id = payload.get("parent_id")
        text = payload.get("text")
        if parent_id is None or not isinstance(text, str):
            return None

        metadata = self._payload_metadata(payload, tenant_id, str(parent_id))

        return RetrievedChunk(
            chunk_id=str(point.id),
            text=text,
            score=score,
            metadata=metadata,
            parent_id=str(parent_id),
        )

    def _tenant_filter(self, tenant_id: str) -> models.Filter:
        return models.Filter(
            must=[
                models.FieldCondition(
                    key="tenant_id",
                    match=models.MatchValue(value=tenant_id),
                )
            ]
        )

    def _rrf_score(self, rank: int) -> float:
        return 1.0 / (RRF_K + rank)

    def _first_result_row(self, rows: Any) -> list[Any]:
        first_row = rows[0] if len(rows) > 0 else []
        if hasattr(first_row, "tolist"):
            return list(first_row.tolist())
        return list(first_row)

    def _chunk_tenant_id(self, chunk: Chunk) -> str | None:
        tenant_id = chunk.metadata.get("tenant_id")
        if tenant_id is None:
            nested_metadata = chunk.metadata.get("metadata")
            if isinstance(nested_metadata, dict):
                tenant_id = nested_metadata.get("tenant_id")
        return str(tenant_id) if tenant_id is not None else None

    def _chunk_parent_id(self, chunk: Chunk) -> str | None:
        parent_id = chunk.metadata.get("parent_id")
        if parent_id is None:
            nested_metadata = chunk.metadata.get("metadata")
            if isinstance(nested_metadata, dict):
                parent_id = nested_metadata.get("parent_id")
        return str(parent_id) if parent_id is not None else None

    def _chunk_metadata(self, chunk: Chunk, tenant_id: str, parent_id: str) -> dict[str, Any]:
        metadata = {
            key: value
            for key, value in chunk.metadata.items()
            if key not in {EMBEDDING_METADATA_KEY, "metadata"}
        }
        nested_metadata = chunk.metadata.get("metadata")
        if isinstance(nested_metadata, dict):
            metadata.update(nested_metadata)

        metadata["tenant_id"] = tenant_id
        metadata["parent_id"] = parent_id
        return metadata

    def _payload_tenant_id(self, payload: dict[str, Any]) -> str | None:
        tenant_id = payload.get("tenant_id")
        if tenant_id is None:
            metadata = payload.get("metadata")
            if isinstance(metadata, dict):
                tenant_id = metadata.get("tenant_id")
        return str(tenant_id) if tenant_id is not None else None

    def _payload_metadata(
        self,
        payload: dict[str, Any],
        tenant_id: str,
        parent_id: str | None = None,
    ) -> dict[str, Any]:
        metadata = payload.get("metadata")
        if not isinstance(metadata, dict):
            metadata = {}
        normalized = dict(metadata)
        normalized["tenant_id"] = tenant_id
        if parent_id is not None:
            normalized["parent_id"] = parent_id

        for key in ("chunk_type", "token_count"):
            if key in payload:
                normalized[key] = payload[key]
        return normalized


class ParentExpander:
    def __init__(
        self,
        tenant_id: str,
        client: QdrantClient | None = None,
        collection_name: str | None = None,
    ) -> None:
        self.tenant_id = tenant_id
        self.client = client or QdrantClient(
            url=settings.QDRANT_URL,
            api_key=settings.QDRANT_API_KEY,
        )
        base_collection_name = collection_name or settings.QDRANT_COLLECTION
        self.parents_collection_name = f"{base_collection_name}_parents"

    def expand(self, children: list[RetrievedChunk]) -> list[ExpandedChunk]:
        start_time = time.perf_counter()
        tenant_id = self._tenant_id_from_children(children)
        tenant_children = [
            child
            for child in children
            if str(child.metadata.get("tenant_id", tenant_id)) == tenant_id
        ]
        parent_ids = self._unique_parent_ids(tenant_children)
        if not parent_ids:
            return []

        records, _next_page = self.client.scroll(
            collection_name=self.parents_collection_name,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="tenant_id",
                        match=models.MatchValue(value=tenant_id),
                    ),
                    models.HasIdCondition(has_id=parent_ids),
                ]
            ),
            limit=len(parent_ids),
            with_payload=True,
            with_vectors=False,
        )
        parent_text_by_id = self._parent_text_by_id(records, tenant_id)

        expanded: list[ExpandedChunk] = []
        seen_parent_ids: set[str] = set()
        for child in tenant_children:
            if child.parent_id in seen_parent_ids:
                continue

            parent_text = parent_text_by_id.get(child.parent_id)
            if parent_text is None:
                continue

            seen_parent_ids.add(child.parent_id)
            expanded.append(
                ExpandedChunk(
                    chunk_id=child.chunk_id,
                    text=child.text,
                    score=child.score,
                    metadata=child.metadata,
                    parent_id=child.parent_id,
                    parent_text=parent_text,
                )
            )

        logger.debug(
            "parent expansion completed",
            extra={
                "tenant_id": tenant_id,
                "latency_seconds": round(time.perf_counter() - start_time, 4),
                "candidate_count": len(tenant_children),
                "expanded_count": len(expanded),
            },
        )
        return expanded

    def _tenant_id_from_children(self, children: list[RetrievedChunk]) -> str:
        for child in children:
            tenant_id = child.metadata.get("tenant_id")
            if tenant_id is not None:
                return str(tenant_id)
        return self.tenant_id

    def _unique_parent_ids(self, children: list[RetrievedChunk]) -> list[str]:
        parent_ids: list[str] = []
        seen_parent_ids: set[str] = set()
        for child in children:
            if child.parent_id in seen_parent_ids:
                continue
            seen_parent_ids.add(child.parent_id)
            parent_ids.append(child.parent_id)
        return parent_ids

    def _parent_text_by_id(self, records: list[Any], tenant_id: str) -> dict[str, str]:
        parent_text_by_id: dict[str, str] = {}
        for record in records:
            payload = record.payload or {}
            if self._payload_tenant_id(payload) != tenant_id:
                continue

            text = payload.get("text")
            if isinstance(text, str):
                parent_text_by_id[str(record.id)] = text
        return parent_text_by_id

    def _payload_tenant_id(self, payload: dict[str, Any]) -> str | None:
        tenant_id = payload.get("tenant_id")
        if tenant_id is None:
            metadata = payload.get("metadata")
            if isinstance(metadata, dict):
                tenant_id = metadata.get("tenant_id")
        return str(tenant_id) if tenant_id is not None else None
