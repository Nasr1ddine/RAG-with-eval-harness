from __future__ import annotations

import hashlib
import json
import math
import os
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any
from uuid import NAMESPACE_URL, uuid4, uuid5

from eval.datasets.schema import EvalSample

GROUNDED_SYSTEM_PROMPT = (
    "Given this document excerpt, generate a factual question that can be answered ONLY "
    "from this excerpt. Return JSON: {question, answer, chunk_ids}"
)
ADVERSARIAL_SYSTEM_PROMPT = (
    "Given this document excerpt, generate a plausible-sounding question about the same "
    "topic that cannot be answered from this excerpt. Return JSON: {question, answer, "
    "chunk_ids}. Use an empty array for chunk_ids."
)
NO_ANSWER_TEXT = "I don't have information on this in the provided documents."
UNANSWERABLE_CATEGORIES = {"adversarial", "no-answer"}


@dataclass(frozen=True)
class _SourceChunk:
    id: str
    text: str
    metadata: dict[str, Any]


class SyntheticDatasetGenerator:
    def __init__(
        self,
        documents: Sequence[Any],
        *,
        llm_client: Any | None = None,
        embedding_client: Any | None = None,
        llm_model: str | None = None,
        embedding_model: str | None = None,
        similarity_threshold: float = 0.92,
        adversarial_ratio: float = 0.2,
        chunk_size_chars: int = 2_400,
        random_seed: int | None = None,
        max_attempts_per_sample: int = 10,
    ) -> None:
        if not documents:
            raise ValueError("documents must contain at least one document")
        if chunk_size_chars <= 0:
            raise ValueError("chunk_size_chars must be positive")
        if not 0.0 <= adversarial_ratio <= 1.0:
            raise ValueError("adversarial_ratio must be between 0.0 and 1.0")

        self.chunks = self._normalize_documents(documents, chunk_size_chars)
        self.llm_client = llm_client
        self.embedding_client = embedding_client
        self.llm_model = llm_model or os.getenv("LLM_MODEL", "gpt-4o-mini")
        self.embedding_model = embedding_model or os.getenv(
            "EMBEDDING_MODEL",
            "text-embedding-3-small",
        )
        self.similarity_threshold = similarity_threshold
        self.adversarial_ratio = adversarial_ratio
        self.random = random.Random(random_seed)
        self.max_attempts_per_sample = max_attempts_per_sample

    def generate(self, n_samples: int, categories: list[str]) -> list[EvalSample]:
        if n_samples < 0:
            raise ValueError("n_samples must be non-negative")
        if n_samples == 0:
            return []

        cleaned_categories = [category.strip() for category in categories if category.strip()]
        if not cleaned_categories:
            cleaned_categories = ["factual"]
        if "adversarial" not in cleaned_categories:
            cleaned_categories.append("adversarial")

        category_plan = self._category_plan(n_samples, cleaned_categories)
        samples: list[EvalSample] = []
        query_embeddings: list[list[float]] = []
        max_attempts = max(n_samples, n_samples * self.max_attempts_per_sample)
        attempts = 0

        while len(samples) < n_samples and attempts < max_attempts:
            attempts += 1
            category = category_plan[len(samples)]
            chunk = self.random.choice(self.chunks)
            candidate = (
                self._generate_unanswerable_sample(chunk, category)
                if category in UNANSWERABLE_CATEGORIES
                else self._generate_grounded_sample(chunk, category)
            )

            if self._is_near_duplicate(candidate.query, query_embeddings):
                continue
            samples.append(candidate)

        if len(samples) < n_samples:
            raise RuntimeError(
                f"Generated {len(samples)} unique samples after {attempts} attempts; "
                "try more source documents or a higher similarity_threshold"
            )
        return samples

    def _generate_grounded_sample(self, chunk: _SourceChunk, category: str) -> EvalSample:
        payload = self._generate_json(
            system_prompt=GROUNDED_SYSTEM_PROMPT,
            user_prompt=self._grounded_user_prompt(chunk, category),
        )
        query = _required_payload_str(payload, "question")
        answer = _required_payload_str(payload, "answer")
        chunk_ids = _payload_chunk_ids(payload)
        if not chunk_ids or chunk.id not in chunk_ids:
            chunk_ids = [chunk.id]

        return EvalSample(
            id=str(uuid4()),
            query=query,
            ground_truth_answer=answer,
            relevant_chunk_ids=chunk_ids,
            tenant_id=self._tenant_id(chunk),
            category=category,
            metadata=self._sample_metadata(chunk),
        )

    def _generate_unanswerable_sample(self, chunk: _SourceChunk, category: str) -> EvalSample:
        payload = self._generate_json(
            system_prompt=ADVERSARIAL_SYSTEM_PROMPT,
            user_prompt=self._adversarial_user_prompt(chunk, category),
        )
        query = _required_payload_str(payload, "question")

        return EvalSample(
            id=str(uuid4()),
            query=query,
            ground_truth_answer=NO_ANSWER_TEXT,
            relevant_chunk_ids=[],
            tenant_id=self._tenant_id(chunk),
            category=category,
            metadata={
                **self._sample_metadata(chunk),
                "unanswerable": True,
            },
        )

    def _generate_json(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        response = self._chat_completion(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )
        content = _message_content(response)
        return _json_object_from_content(content)

    def _chat_completion(self, system_prompt: str, user_prompt: str) -> Any:
        client = self._llm_client()
        kwargs: dict[str, Any] = {
            "model": self.llm_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.0,
            "max_tokens": 500,
        }
        try:
            return client.chat.completions.create(
                **kwargs,
                response_format={"type": "json_object"},
            )
        except TypeError:
            return client.chat.completions.create(**kwargs)

    def _is_near_duplicate(
        self,
        query: str,
        accepted_embeddings: list[list[float]],
    ) -> bool:
        query_embedding = self._embed_query(query)
        if any(
            _cosine_similarity(query_embedding, embedding) > self.similarity_threshold
            for embedding in accepted_embeddings
        ):
            return True
        accepted_embeddings.append(query_embedding)
        return False

    def _embed_query(self, query: str) -> list[float]:
        response = self._embedding_client().embeddings.create(
            model=self.embedding_model,
            input=query,
        )
        embedding = response.data[0].embedding
        if not isinstance(embedding, list):
            raise ValueError("Embedding response must contain a list embedding")
        return [float(value) for value in embedding]

    def _category_plan(self, n_samples: int, categories: list[str]) -> list[str]:
        adversarial_count = max(1, round(n_samples * self.adversarial_ratio))
        adversarial_count = min(adversarial_count, n_samples)
        non_adversarial_categories = [
            category for category in categories if category != "adversarial"
        ]
        if not non_adversarial_categories:
            non_adversarial_categories = ["factual"]

        plan = ["adversarial"] * adversarial_count
        while len(plan) < n_samples:
            index = len(plan) - adversarial_count
            plan.append(non_adversarial_categories[index % len(non_adversarial_categories)])
        self.random.shuffle(plan)
        return plan

    def _llm_client(self) -> Any:
        if self.llm_client is None:
            self.llm_client = self._default_openai_client()
        return self.llm_client

    def _embedding_client(self) -> Any:
        if self.embedding_client is None:
            self.embedding_client = self._default_openai_client()
        return self.embedding_client

    def _default_openai_client(self) -> Any:
        from openai import OpenAI

        return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def _grounded_user_prompt(self, chunk: _SourceChunk, category: str) -> str:
        return (
            f"Category: {category}\n"
            f"Chunk ID: {chunk.id}\n"
            "Document excerpt:\n"
            f"{chunk.text}\n\n"
            "Return a JSON object where chunk_ids contains only the Chunk ID above."
        )

    def _adversarial_user_prompt(self, chunk: _SourceChunk, category: str) -> str:
        return (
            f"Category: {category}\n"
            f"Related chunk ID, for topic only: {chunk.id}\n"
            "Document excerpt:\n"
            f"{chunk.text}\n\n"
            "Return a JSON object with a plausible unanswerable question, "
            f"answer set to {json.dumps(NO_ANSWER_TEXT)}, and chunk_ids set to []."
        )

    def _tenant_id(self, chunk: _SourceChunk) -> str:
        value = chunk.metadata.get("tenant_id", "default")
        return str(value)

    def _sample_metadata(self, chunk: _SourceChunk) -> dict[str, Any]:
        return {
            "source_chunk_id": chunk.id,
            "source_metadata": dict(chunk.metadata),
        }

    def _normalize_documents(
        self,
        documents: Sequence[Any],
        chunk_size_chars: int,
    ) -> list[_SourceChunk]:
        chunks: list[_SourceChunk] = []
        for document_index, document in enumerate(documents):
            text = _document_text(document)
            metadata = _document_metadata(document)
            explicit_chunk_id = _document_id(document, metadata)
            if explicit_chunk_id is not None:
                chunks.append(_SourceChunk(id=explicit_chunk_id, text=text, metadata=metadata))
                continue
            chunks.extend(self._split_document(text, metadata, document_index, chunk_size_chars))

        if not chunks:
            raise ValueError("documents did not contain any text")
        return chunks

    def _split_document(
        self,
        text: str,
        metadata: dict[str, Any],
        document_index: int,
        chunk_size_chars: int,
    ) -> list[_SourceChunk]:
        stripped_text = text.strip()
        if not stripped_text:
            return []

        overlap = min(200, max(0, chunk_size_chars // 10))
        step = max(1, chunk_size_chars - overlap)
        chunks: list[_SourceChunk] = []
        for chunk_index, start in enumerate(range(0, len(stripped_text), step)):
            chunk_text = stripped_text[start : start + chunk_size_chars].strip()
            if not chunk_text:
                continue
            chunk_metadata = {
                **metadata,
                "document_index": document_index,
                "chunk_index": chunk_index,
            }
            chunks.append(
                _SourceChunk(
                    id=_stable_chunk_id(chunk_text, metadata, document_index, chunk_index),
                    text=chunk_text,
                    metadata=chunk_metadata,
                )
            )
        return chunks


def _document_text(document: Any) -> str:
    value = _document_value(document, "text")
    if not isinstance(value, str):
        raise ValueError("Each document must provide a string text field")
    return value


def _document_metadata(document: Any) -> dict[str, Any]:
    value = _document_value(document, "metadata", default={})
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError("Document metadata must be an object")
    return dict(value)


def _document_id(document: Any, metadata: Mapping[str, Any]) -> str | None:
    value = _document_value(document, "chunk_id", default=None)
    if value is None:
        value = _document_value(document, "id", default=None)
    if value is None:
        value = metadata.get("chunk_id") or metadata.get("id")
    return str(value) if value is not None else None


def _document_value(document: Any, key: str, default: Any | None = None) -> Any:
    if isinstance(document, Mapping):
        return document.get(key, default)
    return getattr(document, key, default)


def _stable_chunk_id(
    chunk_text: str,
    metadata: Mapping[str, Any],
    document_index: int,
    chunk_index: int,
) -> str:
    source = str(metadata.get("source", document_index))
    digest = hashlib.sha256(chunk_text.encode("utf-8")).hexdigest()
    return str(uuid5(NAMESPACE_URL, f"{source}:{document_index}:{chunk_index}:{digest}"))


def _required_payload_str(payload: Mapping[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"LLM response must include a non-empty {key} string")
    return value.strip()


def _payload_chunk_ids(payload: Mapping[str, Any]) -> list[str]:
    value = payload.get("chunk_ids", [])
    if not isinstance(value, list):
        return []
    return [str(chunk_id) for chunk_id in value if isinstance(chunk_id, str)]


def _message_content(response: Any) -> str:
    try:
        content = response.choices[0].message.content
    except (AttributeError, IndexError, KeyError, TypeError) as exc:
        raise ValueError("LLM response did not contain message content") from exc
    if not isinstance(content, str) or not content.strip():
        raise ValueError("LLM response content was empty")
    return content


def _json_object_from_content(content: str) -> dict[str, Any]:
    stripped = content.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        stripped = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:]).strip()

    payload = json.loads(stripped)
    if not isinstance(payload, dict):
        raise ValueError("LLM response JSON must be an object")
    return payload


def _cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        raise ValueError("Embedding vectors must have the same dimensions")
    dot_product = sum(left_value * right_value for left_value, right_value in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot_product / (left_norm * right_norm)
