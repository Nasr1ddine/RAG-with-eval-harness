from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from sentence_transformers import CrossEncoder

from services.reranker.config import settings

DEFAULT_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

_model: Any | None = None
_model_name = settings.RERANKER_MODEL or DEFAULT_MODEL_NAME


def model_name() -> str:
    return _model_name


def load_model() -> Any:
    global _model, _model_name

    if _model is None:
        _model_name = settings.RERANKER_MODEL or DEFAULT_MODEL_NAME
        _model = CrossEncoder(_model_name)
        _model.predict([("warmup query", "warmup candidate")])

    return _model


def score_candidates(query: str, candidates: Sequence[str]) -> list[float]:
    if not candidates:
        return []

    model = load_model()
    raw_scores = model.predict([(query, candidate) for candidate in candidates])
    if hasattr(raw_scores, "tolist"):
        values = raw_scores.tolist()
    else:
        values = list(raw_scores)

    return [float(value) for value in values]
