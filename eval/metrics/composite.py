from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class EvalResult:
    sample_id: str
    context_recall: float
    context_precision: float
    faithfulness: float
    answer_relevance: float
    latency_ms: float
    cache_hit: bool
    category: str


MetricSummary = dict[str, float | int | bool]
AggregateResult = dict[str, MetricSummary | dict[str, MetricSummary]]


def aggregate(results: list[EvalResult]) -> AggregateResult:
    """Return overall and per-category means for evaluation results."""
    by_category: dict[str, list[EvalResult]] = {}
    for result in results:
        by_category.setdefault(result.category, []).append(result)

    return {
        "overall": _summarize(results),
        "by_category": {
            category: _summarize(category_results)
            for category, category_results in sorted(by_category.items())
        },
    }


def _summarize(results: list[EvalResult]) -> MetricSummary:
    if not results:
        return {
            "count": 0,
            "context_recall": 0.0,
            "context_precision": 0.0,
            "faithfulness": 0.0,
            "answer_relevance": 0.0,
            "latency_ms": 0.0,
            "cache_hit_rate": 0.0,
            "hallucination_risk": False,
        }

    count = len(results)
    return {
        "count": count,
        "context_recall": sum(result.context_recall for result in results) / count,
        "context_precision": sum(result.context_precision for result in results) / count,
        "faithfulness": sum(result.faithfulness for result in results) / count,
        "answer_relevance": sum(result.answer_relevance for result in results) / count,
        "latency_ms": sum(result.latency_ms for result in results) / count,
        "cache_hit_rate": sum(1.0 for result in results if result.cache_hit) / count,
        "hallucination_risk": any(result.faithfulness < 0.7 for result in results),
    }
