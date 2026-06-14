from __future__ import annotations

import math

from services.api.config import settings
from services.api.pipeline import QueryResponse

LOW_CONFIDENCE_REFUSAL = "i don't have information on this in the provided documents"


def build_answer_eval(response: QueryResponse) -> dict[str, object]:
    source_count = len(response.sources)
    has_context = bool(response.context.strip())
    refused = response.answer.strip().lower().startswith(LOW_CONFIDENCE_REFUSAL)
    source_quality = _source_quality([source.score for source in response.sources])
    source_coverage = min(1.0, source_count / max(1, settings.RETRIEVAL_TOP_K))
    context_quality = 1.0 if has_context else 0.0
    rerank_quality = _rerank_quality(response)

    if refused or not has_context or source_count == 0:
        accuracy = 0.15 if refused else 0.05
    else:
        accuracy = (
            0.45 * source_quality
            + 0.25 * source_coverage
            + 0.15 * context_quality
            + 0.15 * rerank_quality
        )
        if response.cache_hit:
            accuracy = max(accuracy, min(0.95, source_quality))

    accuracy = _clamp_score(accuracy)
    status, status_detail = _accuracy_status(accuracy)

    return {
        "accuracy": accuracy,
        "status": status,
        "status_detail": status_detail,
        "metrics": {
            "source_count": source_count,
            "retrieval_count": response.retrieval_count,
            "reranked_count": response.reranked_count,
            "context_tokens": response.context_tokens,
            "cache_hit": response.cache_hit,
        },
        "components": [
            {"Signal": "Source quality", "Score": round(source_quality, 3)},
            {"Signal": "Source coverage", "Score": round(source_coverage, 3)},
            {"Signal": "Context available", "Score": round(context_quality, 3)},
            {"Signal": "Rerank/cache", "Score": round(rerank_quality, 3)},
        ],
        "sources": _source_score_rows(response),
        "context_preview": _context_preview(response.context),
    }


def _source_quality(scores: list[float]) -> float:
    if not scores:
        return 0.0

    top_score = max(scores)
    if top_score <= 0.0:
        return 0.1
    if top_score <= 0.05:
        return _clamp_score(top_score / 0.02)
    if top_score <= 1.0:
        return _clamp_score(top_score)
    return _clamp_score(1.0 / (1.0 + math.exp(-top_score)))


def _rerank_quality(response: QueryResponse) -> float:
    if response.cache_hit:
        return 1.0
    if response.retrieval_count <= 0:
        return 0.0
    if response.reranked_count <= 0:
        return 0.5
    return _clamp_score(response.reranked_count / response.retrieval_count)


def _source_score_rows(response: QueryResponse) -> list[dict[str, str | float]]:
    rows: list[dict[str, str | float]] = []
    for index, source in enumerate(response.sources, start=1):
        rows.append(
            {
                "Source": f"{index}: {source.chunk_id[:8]}",
                "Chunk ID": source.chunk_id,
                "Raw score": round(source.score, 6),
                "Normalized score": round(_source_quality([source.score]), 3),
            }
        )
    return rows


def _context_preview(context: str, max_chars: int = 1000) -> str:
    cleaned = context.strip()
    if len(cleaned) <= max_chars:
        return cleaned
    return f"{cleaned[:max_chars].rstrip()}..."


def _accuracy_status(accuracy: float) -> tuple[str, str]:
    if accuracy >= 0.75:
        return "High", "Strong retrieval grounding"
    if accuracy >= 0.5:
        return "Review", "Moderate grounding; verify important details"
    return "Low", "Weak or missing grounding"


def _clamp_score(score: float) -> float:
    return max(0.0, min(1.0, score))
