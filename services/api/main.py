from __future__ import annotations

import atexit

import streamlit as st

from services.api.config import settings
from services.api.errors import ServiceUnavailableError
from services.api.pipeline import QueryRequest, QueryResponse
from services.api.runtime import (
    AsyncRuntimeRunner,
    close_sync_runtime_runner,
    get_sync_runtime_runner,
)

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}


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

    _render_upload_section(runtime, tenant_id)
    st.divider()
    _render_chat_section(runtime, tenant_id)


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
            st.markdown(message["content"])
            if message.get("meta"):
                st.caption(message["meta"])

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
            st.markdown(answer)
            st.caption(meta)
            st.session_state.messages.append({"role": "assistant", "content": answer, "meta": meta})


def _format_meta(response: QueryResponse) -> str:
    source_count = len(response.sources)
    parts = [f"{response.latency_ms}ms", f"{source_count} sources"]
    if response.cache_hit:
        parts.append("cache hit")
    return " · ".join(parts)


if __name__ == "__main__":
    main()
