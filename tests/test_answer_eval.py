from __future__ import annotations

import os

os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("QDRANT_URL", "http://qdrant:6333")
os.environ.setdefault("REDIS_URL", "redis://redis:6379")

from services.api.answer_eval import build_answer_eval
from services.api.pipeline import QueryResponse, QuerySource


def test_answer_eval_rates_grounded_answer_high() -> None:
    response = QueryResponse(
        answer="Acme support runs Monday through Friday [chunk-1].",
        sources=[
            QuerySource(chunk_id="chunk-1", score=0.92, text="support hours"),
            QuerySource(chunk_id="chunk-2", score=0.74, text="priority support"),
        ],
        context="[chunk-1]\nAcme support runs Monday through Friday.",
        cache_hit=False,
        latency_ms=1200,
        retrieval_count=4,
        reranked_count=4,
        context_tokens=120,
    )

    answer_eval = build_answer_eval(response)

    assert answer_eval["accuracy"] >= 0.75
    assert answer_eval["status"] == "High"
    assert answer_eval["sources"]


def test_answer_eval_rates_missing_context_low() -> None:
    response = QueryResponse(
        answer="I don't have information on this in the provided documents.",
        sources=[],
        context="",
        cache_hit=False,
        latency_ms=100,
        retrieval_count=0,
        reranked_count=0,
        context_tokens=0,
    )

    answer_eval = build_answer_eval(response)

    assert answer_eval["accuracy"] < 0.5
    assert answer_eval["status"] == "Low"


def test_answer_eval_marks_cache_hit_as_strong_rerank_signal() -> None:
    response = QueryResponse(
        answer="Cached answer.",
        sources=[QuerySource(chunk_id="chunk-1", score=0.95, text="")],
        context="[chunk-1]\nCached context.",
        cache_hit=True,
        latency_ms=20,
        retrieval_count=0,
        reranked_count=0,
        context_tokens=0,
    )

    answer_eval = build_answer_eval(response)
    components = {row["Signal"]: row["Score"] for row in answer_eval["components"]}

    assert components["Rerank/cache"] == 1.0
