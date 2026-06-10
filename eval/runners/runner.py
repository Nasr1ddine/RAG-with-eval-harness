from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import httpx
from rich.console import Console
from rich.table import Table

from eval.datasets.schema import EvalSample
from eval.metrics.composite import EvalResult
from eval.metrics.generation import answer_relevance, faithfulness
from eval.metrics.retrieval import context_precision, context_recall

MIN_CONTEXT_RECALL = 0.75
MIN_CONTEXT_PRECISION = 0.70
MIN_FAITHFULNESS = 0.80
MIN_ANSWER_RELEVANCE = 0.75

HALLUCINATION_RISK_FAITHFULNESS = 0.70

METRIC_KEYS = (
    "context_recall",
    "context_precision",
    "faithfulness",
    "answer_relevance",
)


@dataclass(frozen=True)
class EvalReport:
    run_id: str
    timestamp: str
    dataset_path: str
    n_samples: int
    mean_context_recall: float
    mean_context_precision: float
    mean_faithfulness: float
    mean_answer_relevance: float
    by_category: dict[str, dict[str, float | int]]
    hallucination_risk_count: int
    samples: list[EvalResult]
    passed: bool

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["samples"] = [asdict(sample) for sample in self.samples]
        return payload


class EvalRunner:
    def __init__(
        self,
        api_base_url: str,
        dataset_path: str,
        output_dir: str,
        parallelism: int = 4,
    ) -> None:
        if parallelism < 1:
            raise ValueError("parallelism must be at least 1")

        self.api_base_url = api_base_url.rstrip("/")
        self.dataset_path = dataset_path
        self.output_dir = Path(output_dir)
        self.parallelism = parallelism
        self.console = Console()

    async def run_eval(self, dataset: list[EvalSample]) -> EvalReport:
        semaphore = asyncio.Semaphore(self.parallelism)

        async with httpx.AsyncClient(base_url=self.api_base_url, timeout=60.0) as client:
            tasks = [self._evaluate_sample(sample, client, semaphore) for sample in dataset]
            results = list(await asyncio.gather(*tasks))

        report = self._build_report(results)
        self._save_report(report)
        self._print_summary(report)
        return report

    async def _evaluate_sample(
        self,
        sample: EvalSample,
        client: httpx.AsyncClient,
        semaphore: asyncio.Semaphore,
    ) -> EvalResult:
        async with semaphore:
            response = await client.post(
                "/query",
                json={"query": sample.query, "tenant_id": sample.tenant_id},
            )
            response.raise_for_status()
            payload = response.json()

            retrieved_chunk_ids, scores, context = _extract_retrieval_payload(payload, sample)
            recall = context_recall(retrieved_chunk_ids, sample.relevant_chunk_ids)
            precision = context_precision(retrieved_chunk_ids, sample.relevant_chunk_ids, scores)

            answer = _string_value(payload, "answer")
            faithfulness_score, relevance_score = await asyncio.gather(
                faithfulness(answer, context),
                answer_relevance(sample.query, answer),
            )

            return EvalResult(
                sample_id=sample.id,
                context_recall=recall,
                context_precision=precision,
                faithfulness=faithfulness_score,
                answer_relevance=relevance_score,
                latency_ms=float(payload.get("latency_ms", 0.0)),
                cache_hit=bool(payload.get("cache_hit", False)),
                category=sample.category,
            )

    def _build_report(self, results: list[EvalResult]) -> EvalReport:
        timestamp = datetime.now(UTC).isoformat()
        overall = _summarize(results)
        by_category = _summarize_by_category(results)
        passed = (
            overall["context_recall"] >= MIN_CONTEXT_RECALL
            and overall["context_precision"] >= MIN_CONTEXT_PRECISION
            and overall["faithfulness"] >= MIN_FAITHFULNESS
            and overall["answer_relevance"] >= MIN_ANSWER_RELEVANCE
        )

        return EvalReport(
            run_id=str(uuid4()),
            timestamp=timestamp,
            dataset_path=self.dataset_path,
            n_samples=len(results),
            mean_context_recall=float(overall["context_recall"]),
            mean_context_precision=float(overall["context_precision"]),
            mean_faithfulness=float(overall["faithfulness"]),
            mean_answer_relevance=float(overall["answer_relevance"]),
            by_category=by_category,
            hallucination_risk_count=sum(
                1 for result in results if result.faithfulness < HALLUCINATION_RISK_FAITHFULNESS
            ),
            samples=results,
            passed=passed,
        )

    def _save_report(self, report: EvalReport) -> Path:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        filename_timestamp = _filename_timestamp(report.timestamp)
        output_path = self.output_dir / f"eval_{filename_timestamp}.json"
        output_path.write_text(
            json.dumps(report.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return output_path

    def _print_summary(self, report: EvalReport) -> None:
        table = Table(title=f"RAG Eval Summary ({'PASS' if report.passed else 'FAIL'})")
        table.add_column("Scope")
        table.add_column("Samples", justify="right")
        table.add_column("Context Recall", justify="right")
        table.add_column("Context Precision", justify="right")
        table.add_column("Faithfulness", justify="right")
        table.add_column("Answer Relevance", justify="right")
        table.add_column("Hallucination Risk", justify="right")

        table.add_row(
            "overall",
            str(report.n_samples),
            _format_score(report.mean_context_recall),
            _format_score(report.mean_context_precision),
            _format_score(report.mean_faithfulness),
            _format_score(report.mean_answer_relevance),
            str(report.hallucination_risk_count),
        )

        for category, summary in report.by_category.items():
            table.add_row(
                category,
                str(summary["count"]),
                _format_score(float(summary["context_recall"])),
                _format_score(float(summary["context_precision"])),
                _format_score(float(summary["faithfulness"])),
                _format_score(float(summary["answer_relevance"])),
                str(summary["hallucination_risk_count"]),
            )

        self.console.print(table)


def _extract_retrieval_payload(
    payload: Mapping[str, Any],
    sample: EvalSample,
) -> tuple[list[str], list[float], str]:
    source_items = _source_items(payload)
    retrieved_chunk_ids = _retrieved_chunk_ids(payload, source_items)
    scores = _scores(source_items, len(retrieved_chunk_ids))
    context = _context(payload, source_items, sample)
    return retrieved_chunk_ids, scores, context


def _source_items(payload: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    raw_sources = (
        payload.get("sources")
        or payload.get("retrieved_chunks")
        or payload.get("chunks")
        or []
    )
    if not isinstance(raw_sources, Sequence) or isinstance(raw_sources, str):
        return []
    return [source for source in raw_sources if isinstance(source, Mapping)]


def _retrieved_chunk_ids(
    payload: Mapping[str, Any],
    source_items: Sequence[Mapping[str, Any]],
) -> list[str]:
    raw_ids = payload.get("retrieved_chunk_ids")
    if isinstance(raw_ids, Sequence) and not isinstance(raw_ids, str):
        return [str(chunk_id) for chunk_id in raw_ids]

    chunk_ids: list[str] = []
    for source in source_items:
        chunk_id = source.get("chunk_id") or source.get("id")
        if chunk_id is not None:
            chunk_ids.append(str(chunk_id))
    return chunk_ids


def _scores(source_items: Sequence[Mapping[str, Any]], chunk_count: int) -> list[float]:
    scores: list[float] = []
    for source in source_items:
        raw_score = source.get("score")
        scores.append(float(raw_score) if isinstance(raw_score, int | float) else 0.0)

    if len(scores) == chunk_count:
        return scores

    # Preserve API order if explicit scores are unavailable.
    return [float(chunk_count - index) for index in range(chunk_count)]


def _context(
    payload: Mapping[str, Any],
    source_items: Sequence[Mapping[str, Any]],
    sample: EvalSample,
) -> str:
    payload_context = payload.get("context")
    if isinstance(payload_context, str) and payload_context.strip():
        return payload_context

    source_texts = [
        str(text)
        for source in source_items
        for text in (source.get("text"), source.get("content"))
        if isinstance(text, str) and text.strip()
    ]
    if source_texts:
        return "\n\n".join(source_texts)

    for key in ("context", "source_text", "source_context", "expected_context"):
        metadata_value = sample.metadata.get(key)
        if isinstance(metadata_value, str) and metadata_value.strip():
            return metadata_value

    return (
        "Reference answer:\n"
        f"{sample.ground_truth_answer}\n\n"
        "Retrieved chunk IDs:\n"
        f"{', '.join(_retrieved_chunk_ids(payload, source_items))}"
    )


def _string_value(payload: Mapping[str, Any], key: str) -> str:
    value = payload.get(key)
    return value if isinstance(value, str) else ""


def _summarize(results: list[EvalResult]) -> dict[str, float | int]:
    count = len(results)
    if count == 0:
        return {
            "count": 0,
            "context_recall": 0.0,
            "context_precision": 0.0,
            "faithfulness": 0.0,
            "answer_relevance": 0.0,
            "latency_ms": 0.0,
            "cache_hit_rate": 0.0,
            "hallucination_risk_count": 0,
        }

    return {
        "count": count,
        "context_recall": sum(result.context_recall for result in results) / count,
        "context_precision": sum(result.context_precision for result in results) / count,
        "faithfulness": sum(result.faithfulness for result in results) / count,
        "answer_relevance": sum(result.answer_relevance for result in results) / count,
        "latency_ms": sum(result.latency_ms for result in results) / count,
        "cache_hit_rate": sum(1.0 for result in results if result.cache_hit) / count,
        "hallucination_risk_count": sum(
            1 for result in results if result.faithfulness < HALLUCINATION_RISK_FAITHFULNESS
        ),
    }


def _summarize_by_category(results: list[EvalResult]) -> dict[str, dict[str, float | int]]:
    grouped: dict[str, list[EvalResult]] = {}
    for result in results:
        grouped.setdefault(result.category, []).append(result)

    return {
        category: _summarize(category_results)
        for category, category_results in sorted(grouped.items())
    }


def _filename_timestamp(timestamp: str) -> str:
    return timestamp.replace("+00:00", "Z").replace(":", "-").replace(".", "-")


def _format_score(score: float) -> str:
    return f"{score:.3f}"
