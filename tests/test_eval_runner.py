from eval.datasets.schema import EvalSample
from eval.runners.runner import _extract_retrieval_payload


def test_extract_retrieval_payload_prefers_response_context() -> None:
    sample = EvalSample(
        id="sample-1",
        query="What are support hours?",
        ground_truth_answer="Support runs Monday through Friday.",
        relevant_chunk_ids=["chunk-1"],
        tenant_id="default",
        category="support",
        metadata={"expected_context": "fallback context"},
    )
    payload = {
        "sources": [
            {"chunk_id": "chunk-1", "score": 0.9, "text": "source text"},
            {"chunk_id": "chunk-2", "score": 0.1, "text": "other text"},
        ],
        "context": "actual retrieved context",
    }

    chunk_ids, scores, context = _extract_retrieval_payload(payload, sample)

    assert chunk_ids == ["chunk-1", "chunk-2"]
    assert scores == [0.9, 0.1]
    assert context == "actual retrieved context"


def test_extract_retrieval_payload_uses_source_text_when_context_missing() -> None:
    sample = EvalSample(
        id="sample-1",
        query="What are support hours?",
        ground_truth_answer="Support runs Monday through Friday.",
        relevant_chunk_ids=["chunk-1"],
        tenant_id="default",
        category="support",
    )
    payload = {
        "sources": [
            {"chunk_id": "chunk-1", "score": 0.9, "text": "first source"},
            {"chunk_id": "chunk-2", "score": 0.1, "content": "second source"},
        ],
    }

    _chunk_ids, _scores, context = _extract_retrieval_payload(payload, sample)

    assert context == "first source\n\nsecond source"
