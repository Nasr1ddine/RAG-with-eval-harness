# Dokploy Prometheus scrape config:
# scrape_configs:
#   - job_name: "rag-api"
#     metrics_path: /metrics
#     static_configs:
#       - targets: ["api:8000"]

from __future__ import annotations

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, generate_latest

REGISTRY = CollectorRegistry(auto_describe=True)

RAG_REQUESTS_TOTAL = Counter(
    "rag_requests_total",
    "Total RAG query requests.",
    ("tenant_id", "cache_hit"),
    registry=REGISTRY,
)
RAG_LATENCY_MS = Histogram(
    "rag_latency_ms",
    "RAG query latency in milliseconds.",
    buckets=(100, 300, 500, 1000, 2000, 5000),
    registry=REGISTRY,
)
RAG_RETRIEVAL_CANDIDATES = Histogram(
    "rag_retrieval_candidates",
    "Number of retrieval candidates returned before context assembly.",
    buckets=(1, 5, 10, 20, 50, 100),
    registry=REGISTRY,
)
RAG_CONTEXT_TOKENS = Histogram(
    "rag_context_tokens",
    "Number of context tokens sent to the answer model.",
    buckets=(100, 300, 500, 1000, 2000, 4000, 6000, 8000),
    registry=REGISTRY,
)
RAG_FAITHFULNESS_SCORE = Gauge(
    "rag_faithfulness_score",
    "Latest sampled RAG faithfulness score.",
    registry=REGISTRY,
)


def render_metrics() -> bytes:
    return generate_latest(REGISTRY)


def record_rag_request(
    *,
    tenant_id: str,
    cache_hit: bool,
    latency_ms: int,
    retrieval_candidates: int,
    context_tokens: int,
) -> None:
    RAG_REQUESTS_TOTAL.labels(tenant_id=tenant_id, cache_hit=str(cache_hit).lower()).inc()
    RAG_LATENCY_MS.observe(latency_ms)
    RAG_RETRIEVAL_CANDIDATES.observe(retrieval_candidates)
    RAG_CONTEXT_TOKENS.observe(context_tokens)


def update_faithfulness_score(score: float) -> None:
    RAG_FAITHFULNESS_SCORE.set(score)
