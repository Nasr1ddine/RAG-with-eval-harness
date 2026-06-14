from __future__ import annotations

import atexit
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import streamlit as st

from services.api.answer_eval import build_answer_eval
from services.api.config import settings
from services.api.errors import ServiceUnavailableError
from services.api.pipeline import QueryRequest, QueryResponse
from services.api.runtime import (
    AsyncRuntimeRunner,
    close_sync_runtime_runner,
    get_sync_runtime_runner,
)

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}
EVAL_RESULTS_DIR = Path("eval/results")
EVAL_METRICS = (
    ("Context Recall", "mean_context_recall"),
    ("Context Precision", "mean_context_precision"),
    ("Faithfulness", "mean_faithfulness"),
    ("Answer Relevance", "mean_answer_relevance"),
)


@dataclass(frozen=True)
class EvalReportView:
    path: Path
    payload: dict[str, Any]


@st.cache_resource
def _load_runtime() -> AsyncRuntimeRunner:
    runtime = get_sync_runtime_runner()
    atexit.register(close_sync_runtime_runner)
    return runtime


def main() -> None:
    st.set_page_config(page_title="RAG System", page_icon="📄", layout="wide")
    st.title("RAG System")
    st.caption("Upload documents and ask questions grounded in your content.")

    runtime = _load_runtime()

    tenant_id = st.sidebar.text_input("Tenant ID", value=settings.DEFAULT_TENANT_ID)
    tenant_id = tenant_id.strip() or settings.DEFAULT_TENANT_ID

    chat_tab, eval_tab = st.tabs(["Chat", "Evaluations"])
    with chat_tab:
        _render_upload_section(runtime, tenant_id)
        st.divider()
        _render_chat_section(runtime, tenant_id)

    with eval_tab:
        _render_eval_section()


def _render_upload_section(runtime: AsyncRuntimeRunner, tenant_id: str) -> None:
    st.subheader("Upload document")
    st.caption("Supported: PDF, DOCX, TXT, MD")

    uploaded_file = st.file_uploader(
        "Choose a file",
        type=["pdf", "docx", "txt", "md"],
        label_visibility="collapsed",
    )

    if uploaded_file is None:
        return

    name = uploaded_file.name
    suffix = f".{name.rsplit('.', 1)[-1].lower()}" if "." in name else ""
    if suffix and suffix not in SUPPORTED_EXTENSIONS:
        st.error("Unsupported file type.")
        return

    if st.button("Upload", type="primary"):
        with st.spinner("Ingesting document..."):
            try:
                result = runtime.ingest_document(
                    filename=uploaded_file.name,
                    content=uploaded_file.getvalue(),
                    content_type=uploaded_file.type or "application/octet-stream",
                    tenant_id=tenant_id,
                )
            except ServiceUnavailableError as exc:
                st.error(str(exc))
                return

        child_count = result.get("child_count", "?")
        parent_count = result.get("parent_count", "?")
        duration = float(result.get("duration_seconds", 0.0))
        st.success(f"Ingested {child_count} chunks ({parent_count} parents) in {duration:.1f}s.")


def _render_chat_section(runtime: AsyncRuntimeRunner, tenant_id: str) -> None:
    st.subheader("Ask a question")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant":
                _render_assistant_message(message)
            else:
                st.markdown(message["content"])

    if prompt := st.chat_input("Ask something about your uploaded documents..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = runtime.query(QueryRequest(query=prompt, tenant_id=tenant_id))
                except ServiceUnavailableError as exc:
                    answer = f"Error: {exc}"
                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                    return

            answer = response.answer
            meta = _format_meta(response)
            answer_eval = build_answer_eval(response)
            st.markdown(answer)
            _render_answer_eval(answer_eval, meta)
            st.session_state.messages.append(
                {"role": "assistant", "content": answer, "meta": meta, "eval": answer_eval}
            )


def _format_meta(response: QueryResponse) -> str:
    source_count = len(response.sources)
    parts = [f"{response.latency_ms}ms", f"{source_count} sources"]
    if response.cache_hit:
        parts.append("cache hit")
    return " · ".join(parts)


def _render_assistant_message(message: dict[str, Any]) -> None:
    st.markdown(str(message["content"]))
    answer_eval = message.get("eval")
    meta = message.get("meta")
    if isinstance(answer_eval, dict):
        _render_answer_eval(answer_eval, str(meta or ""))
    elif meta:
        st.caption(str(meta))


def _render_answer_eval(answer_eval: dict[str, Any], meta: str) -> None:
    accuracy = _numeric_value(answer_eval, "accuracy")
    status = _string_value(answer_eval, "status")
    status_detail = _string_value(answer_eval, "status_detail")

    summary_columns = st.columns([1, 1, 2])
    summary_columns[0].metric("Estimated accuracy", f"{accuracy:.0%}")
    summary_columns[1].metric("Eval", status)
    summary_columns[2].caption(f"{status_detail} · {meta}" if meta else status_detail)
    st.progress(min(1.0, max(0.0, accuracy)))

    with st.expander("More details"):
        st.caption(
            "This is an estimated grounding score based on retrieval signals, source coverage, "
            "reranking, cache status, and available context. It is not a ground-truth eval."
        )

        metrics = answer_eval.get("metrics", {})
        if isinstance(metrics, dict):
            metric_columns = st.columns(4)
            metric_columns[0].metric("Sources", int(_numeric_value(metrics, "source_count")))
            metric_columns[1].metric("Retrieved", int(_numeric_value(metrics, "retrieval_count")))
            metric_columns[2].metric("Reranked", int(_numeric_value(metrics, "reranked_count")))
            metric_columns[3].metric(
                "Context tokens", int(_numeric_value(metrics, "context_tokens"))
            )

        components = answer_eval.get("components")
        if isinstance(components, list) and components:
            st.markdown("#### Quality Signals")
            st.bar_chart(components, x="Signal", y="Score")
            st.dataframe(components, hide_index=True, use_container_width=True)

        sources = answer_eval.get("sources")
        if isinstance(sources, list) and sources:
            st.markdown("#### Source Scores")
            st.bar_chart(sources, x="Source", y="Normalized score")
            st.dataframe(sources, hide_index=True, use_container_width=True)

        context_preview = _string_value(answer_eval, "context_preview")
        if context_preview:
            st.markdown("#### Context Preview")
            st.code(context_preview, language="text")


def _render_eval_section() -> None:
    st.subheader("Evaluation reports")
    st.caption("Visualizes JSON reports written by `rag-eval run --output-dir eval/results`.")

    reports = _load_eval_reports(EVAL_RESULTS_DIR)
    if not reports:
        st.info("No eval reports found in `eval/results` yet.")
        st.code(
            "uv run rag-eval run --dataset eval/datasets/sample.jsonl --direct "
            "--output-dir eval/results",
            language="powershell",
        )
        return

    selected = st.selectbox(
        "Report",
        options=reports,
        format_func=_eval_report_label,
    )
    if selected is None:
        return

    report = selected.payload
    st.caption(f"Report file: `{selected.path}`")

    status = "PASS" if bool(report.get("passed", False)) else "FAIL"
    status_method = st.success if status == "PASS" else st.error
    status_method(
        f"{status} · {int(_numeric_value(report, 'n_samples'))} samples · "
        f"{_string_value(report, 'timestamp')}"
    )

    metric_columns = st.columns(len(EVAL_METRICS))
    for column, (label, key) in zip(metric_columns, EVAL_METRICS, strict=True):
        column.metric(label, _format_score(_numeric_value(report, key)))

    _render_category_breakdown(report)
    _render_ragas_summary(report)
    _render_sample_results(report)


def _load_eval_reports(results_dir: Path) -> list[EvalReportView]:
    if not results_dir.exists():
        return []

    reports: list[EvalReportView] = []
    for path in sorted(results_dir.glob("*.json"), reverse=True):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict):
            reports.append(EvalReportView(path=path, payload=cast(dict[str, Any], payload)))
    return sorted(reports, key=_eval_report_sort_key, reverse=True)


def _eval_report_sort_key(report: EvalReportView) -> str:
    timestamp = report.payload.get("timestamp")
    if isinstance(timestamp, str):
        return timestamp
    return report.path.name


def _eval_report_label(report: EvalReportView) -> str:
    timestamp = _string_value(report.payload, "timestamp") or report.path.name
    dataset_path = _string_value(report.payload, "dataset_path")
    status = "PASS" if bool(report.payload.get("passed", False)) else "FAIL"
    if dataset_path:
        return f"{timestamp} · {status} · {dataset_path}"
    return f"{timestamp} · {status}"


def _render_category_breakdown(report: dict[str, Any]) -> None:
    by_category = report.get("by_category")
    if not isinstance(by_category, dict) or not by_category:
        return

    rows: list[dict[str, str | int | float]] = []
    for category, summary in sorted(by_category.items()):
        if not isinstance(summary, dict):
            continue
        rows.append(
            {
                "Category": str(category),
                "Samples": int(_numeric_value(summary, "count")),
                "Context Recall": _numeric_value(summary, "context_recall"),
                "Context Precision": _numeric_value(summary, "context_precision"),
                "Faithfulness": _numeric_value(summary, "faithfulness"),
                "Answer Relevance": _numeric_value(summary, "answer_relevance"),
                "Hallucination Risk": int(_numeric_value(summary, "hallucination_risk_count")),
            }
        )

    if rows:
        st.markdown("### By Category")
        st.dataframe(rows, hide_index=True, use_container_width=True)
        st.bar_chart(
            rows,
            x="Category",
            y=["Context Recall", "Context Precision", "Faithfulness", "Answer Relevance"],
        )


def _render_ragas_summary(report: dict[str, Any]) -> None:
    ragas_summary = report.get("ragas_summary")
    if not isinstance(ragas_summary, dict):
        return

    rows = [
        {"Metric": str(metric), "Score": float(score)}
        for metric, score in ragas_summary.items()
        if isinstance(score, int | float)
    ]
    if not rows:
        return

    st.markdown("### RAGAS Scores")
    st.dataframe(rows, hide_index=True, use_container_width=True)
    st.bar_chart(rows, x="Metric", y="Score")


def _render_sample_results(report: dict[str, Any]) -> None:
    samples = report.get("samples")
    if not isinstance(samples, list) or not samples:
        return

    rows: list[dict[str, str | bool | float]] = []
    for sample in samples:
        if not isinstance(sample, dict):
            continue
        rows.append(
            {
                "Sample ID": _string_value(sample, "sample_id"),
                "Category": _string_value(sample, "category"),
                "Context Recall": _numeric_value(sample, "context_recall"),
                "Context Precision": _numeric_value(sample, "context_precision"),
                "Faithfulness": _numeric_value(sample, "faithfulness"),
                "Answer Relevance": _numeric_value(sample, "answer_relevance"),
                "Latency (ms)": _numeric_value(sample, "latency_ms"),
                "Cache Hit": bool(sample.get("cache_hit", False)),
            }
        )

    if rows:
        st.markdown("### Samples")
        st.dataframe(rows, hide_index=True, use_container_width=True)


def _numeric_value(payload: dict[str, Any], key: str) -> float:
    value = payload.get(key)
    if isinstance(value, int | float):
        return float(value)
    return 0.0


def _string_value(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    return value if isinstance(value, str) else ""


def _format_score(score: float) -> str:
    return f"{score:.3f}"


if __name__ == "__main__":
    main()
