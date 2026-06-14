from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from uuid import NAMESPACE_URL, uuid5

import tiktoken


@dataclass(frozen=True)
class Document:
    text: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class Chunk:
    id: str
    text: str
    metadata: dict[str, Any]
    token_count: int


@dataclass(frozen=True)
class ChunkPair:
    parent: Chunk
    children: list[Chunk]


class HierarchicalChunker:
    def __init__(
        self,
        parent_size: int = 1500,
        child_size: int = 300,
        overlap: int = 50,
        encoding_name: str = "cl100k_base",
    ) -> None:
        self.parent_size = parent_size
        self.child_size = child_size
        self.overlap = overlap
        self.encoding = tiktoken.get_encoding(encoding_name)

        if overlap >= parent_size or overlap >= child_size:
            raise ValueError("overlap must be smaller than both chunk sizes")

    def chunk(self, docs: list[Document]) -> list[ChunkPair]:
        chunk_pairs: list[ChunkPair] = []

        for doc in docs:
            source = str(doc.metadata.get("source", "unknown"))
            for parent_index, parent_tokens in enumerate(
                self._token_windows(doc.text, self.parent_size)
            ):
                parent_text = self.encoding.decode(parent_tokens)
                parent = Chunk(
                    id=self._parent_id(source, parent_index),
                    text=parent_text,
                    metadata={
                        **doc.metadata,
                        "chunk_type": "parent",
                        "parent_index": parent_index,
                    },
                    token_count=len(parent_tokens),
                )

                parent_summary = parent_text[:80].replace("\n", " ").strip()
                children = [
                    self._make_child_chunk(
                        child_tokens=child_tokens,
                        doc_metadata=doc.metadata,
                        source=source,
                        parent=parent,
                        child_index=child_index,
                        parent_summary=parent_summary,
                    )
                    for child_index, child_tokens in enumerate(
                        self._token_windows(parent_text, self.child_size)
                    )
                ]
                chunk_pairs.append(ChunkPair(parent=parent, children=children))

        return chunk_pairs

    def _make_child_chunk(
        self,
        child_tokens: list[int],
        doc_metadata: dict[str, Any],
        source: str,
        parent: Chunk,
        child_index: int,
        parent_summary: str,
    ) -> Chunk:
        child_text = self.encoding.decode(child_tokens)
        prefixed_text = f"[Source: {source} | Section: {parent_summary}]\n{child_text}"
        token_count = len(self.encoding.encode(prefixed_text))

        return Chunk(
            id=self._child_id(source, parent.metadata["parent_index"], child_index),
            text=prefixed_text,
            metadata={
                **doc_metadata,
                "chunk_type": "child",
                "parent_id": parent.id,
                "parent_index": parent.metadata["parent_index"],
                "child_index": child_index,
            },
            token_count=token_count,
        )

    def _token_windows(self, text: str, chunk_size: int) -> list[list[int]]:
        tokens = self.encoding.encode(text)
        if not tokens:
            return []

        step = chunk_size - self.overlap
        return [tokens[start : start + chunk_size] for start in range(0, len(tokens), step)]

    def _parent_id(self, source: str, parent_index: int) -> str:
        return str(uuid5(NAMESPACE_URL, f"{source}:parent:{parent_index}"))

    def _child_id(self, source: str, parent_index: Any, child_index: int) -> str:
        return str(uuid5(NAMESPACE_URL, f"{source}:parent:{parent_index}:child:{child_index}"))
