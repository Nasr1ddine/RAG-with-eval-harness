from __future__ import annotations

from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http import models
from services.ingestion.chunker import Chunk
from services.ingestion.config import settings

EMBEDDING_METADATA_KEY = "_embedding"


class QdrantIndexer:
    def __init__(
        self,
        client: QdrantClient | None = None,
        collection_name: str | None = None,
        vector_size: int | None = None,
    ) -> None:
        self.client = client or QdrantClient(
            url=settings.QDRANT_URL,
            api_key=settings.QDRANT_API_KEY,
        )
        self.collection_name = collection_name or settings.QDRANT_COLLECTION
        self.vector_size = vector_size or settings.EMBEDDING_DIMENSIONS

    def upsert_chunks(self, chunks: list[Chunk], tenant_id: str) -> None:
        parent_chunks = [chunk for chunk in chunks if chunk.metadata.get("chunk_type") == "parent"]
        child_chunks = [chunk for chunk in chunks if chunk.metadata.get("chunk_type") == "child"]

        self._upsert_collection(
            collection_name=f"{self.collection_name}_parents",
            chunks=parent_chunks,
            tenant_id=tenant_id,
            batch_size=64,
        )
        self._upsert_collection(
            collection_name=f"{self.collection_name}_children",
            chunks=child_chunks,
            tenant_id=tenant_id,
            batch_size=64,
        )

    def _upsert_collection(
        self,
        collection_name: str,
        chunks: list[Chunk],
        tenant_id: str,
        batch_size: int,
    ) -> None:
        if not chunks:
            return

        self._ensure_collection(collection_name)

        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            points = [
                models.PointStruct(
                    id=chunk.id,
                    vector=self._get_embedding(chunk),
                    payload=self._payload_for_chunk(chunk, tenant_id),
                )
                for chunk in batch
            ]
            self.client.upsert(collection_name=collection_name, points=points)

    def _ensure_collection(self, collection_name: str) -> None:
        if self.client.collection_exists(collection_name):
            return

        self.client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=self.vector_size,
                distance=models.Distance.COSINE,
            ),
        )

    def _get_embedding(self, chunk: Chunk) -> list[float]:
        embedding = chunk.metadata.get(EMBEDDING_METADATA_KEY)
        if not isinstance(embedding, list):
            raise ValueError(f"Chunk {chunk.id} is missing embedding metadata")
        return embedding

    def _payload_for_chunk(self, chunk: Chunk, tenant_id: str) -> dict[str, Any]:
        metadata = {
            key: value
            for key, value in chunk.metadata.items()
            if key != EMBEDDING_METADATA_KEY
        }
        chunk_type = metadata.get("chunk_type")

        return {
            "text": chunk.text,
            "metadata": metadata,
            "tenant_id": tenant_id,
            "chunk_type": chunk_type,
            "parent_id": metadata.get("parent_id"),
            "token_count": chunk.token_count,
        }
