from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Self


@dataclass(frozen=True)
class EvalSample:
    id: str
    query: str
    ground_truth_answer: str
    relevant_chunk_ids: list[str]
    tenant_id: str
    category: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "query": self.query,
            "ground_truth_answer": self.ground_truth_answer,
            "relevant_chunk_ids": list(self.relevant_chunk_ids),
            "tenant_id": self.tenant_id,
            "category": self.category,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Self:
        return cls(
            id=_required_str(payload, "id"),
            query=_required_str(payload, "query"),
            ground_truth_answer=_required_str(payload, "ground_truth_answer"),
            relevant_chunk_ids=_required_str_list(payload, "relevant_chunk_ids"),
            tenant_id=_required_str(payload, "tenant_id"),
            category=_required_str(payload, "category"),
            metadata=_metadata(payload),
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_json(cls, raw: str) -> Self:
        payload = json.loads(raw)
        if not isinstance(payload, Mapping):
            raise ValueError("EvalSample JSON must be an object")
        return cls.from_dict(payload)

    def to_jsonl_line(self) -> str:
        return self.to_json()

    @classmethod
    def from_jsonl_line(cls, raw: str) -> Self:
        return cls.from_json(raw)


def _required_str(payload: Mapping[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str):
        raise ValueError(f"EvalSample.{key} must be a string")
    return value


def _required_str_list(payload: Mapping[str, Any], key: str) -> list[str]:
    value = payload.get(key)
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"EvalSample.{key} must be a list of strings")
    return list(value)


def _metadata(payload: Mapping[str, Any]) -> dict[str, Any]:
    value = payload.get("metadata", {})
    if not isinstance(value, dict):
        raise ValueError("EvalSample.metadata must be an object")
    return dict(value)
