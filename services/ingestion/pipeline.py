from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from unstructured.partition.auto import partition

from services.ingestion.chunker import Document, HierarchicalChunker
from services.ingestion.config import settings
from services.ingestion.embedder import BatchEmbedder
from services.ingestion.indexer import EMBEDDING_METADATA_KEY, QdrantIndexer

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}


@dataclass(frozen=True)
class IngestionResult:
    parent_count: int
    child_count: int
    total_tokens: int
    duration_seconds: float


class IngestionPipeline:
    def __init__(
        self,
        chunker: HierarchicalChunker | None = None,
        embedder: BatchEmbedder | None = None,
        indexer: QdrantIndexer | None = None,
    ) -> None:
        self.chunker = chunker or HierarchicalChunker(
            parent_size=settings.INGESTION_PARENT_CHUNK_SIZE,
            child_size=settings.INGESTION_CHILD_CHUNK_SIZE,
            overlap=settings.INGESTION_CHUNK_OVERLAP,
        )
        self.embedder = embedder or BatchEmbedder()
        self.indexer = indexer or QdrantIndexer()

    async def run(
        self,
        file_path: str,
        metadata: dict[str, Any],
        tenant_id: str,
    ) -> IngestionResult:
        start_time = time.perf_counter()
        path = Path(file_path)
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
            raise ValueError(f"Unsupported document type {path.suffix!r}; supported: {supported}")

        docs = self._load_documents(path, metadata)
        chunk_pairs = self.chunker.chunk(docs)
        parent_chunks = [pair.parent for pair in chunk_pairs]
        child_chunks = [child for pair in chunk_pairs for child in pair.children]
        chunks = [*parent_chunks, *child_chunks]

        embeddings = await self.embedder.embed_batch(
            [chunk.text for chunk in chunks],
            batch_size=settings.INGESTION_BATCH_SIZE,
        )
        for chunk, embedding in zip(chunks, embeddings, strict=True):
            chunk.metadata[EMBEDDING_METADATA_KEY] = embedding

        self.indexer.upsert_chunks(chunks, tenant_id=tenant_id)

        result = IngestionResult(
            parent_count=len(parent_chunks),
            child_count=len(child_chunks),
            total_tokens=sum(chunk.token_count for chunk in chunks),
            duration_seconds=round(time.perf_counter() - start_time, 3),
        )
        print(
            json.dumps(
                {
                    "event": "ingestion_run",
                    "file_path": str(path),
                    "tenant_id": tenant_id,
                    "parent_count": result.parent_count,
                    "child_count": result.child_count,
                    "total_tokens": result.total_tokens,
                    "duration_seconds": result.duration_seconds,
                }
            ),
            flush=True,
        )

        return result

    def _load_documents(self, path: Path, metadata: dict[str, Any]) -> list[Document]:
        elements = partition(filename=str(path))
        text = "\n\n".join(str(element).strip() for element in elements if str(element).strip())
        if not text:
            raise ValueError(f"No text could be extracted from {path}")

        document_metadata = {
            **metadata,
            "source": metadata.get("source", path.name),
            "file_path": str(path),
            "file_name": path.name,
            "file_type": path.suffix.lower().lstrip("."),
        }
        return [Document(text=text, metadata=document_metadata)]
