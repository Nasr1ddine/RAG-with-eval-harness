from __future__ import annotations


def context_recall(retrieved_chunk_ids: list[str], relevant_chunk_ids: list[str]) -> float:
    """Return the share of relevant chunks present in the retrieved set."""
    relevant_ids = set(relevant_chunk_ids)
    if not relevant_ids:
        return 0.0

    retrieved_ids = set(retrieved_chunk_ids)
    return len(retrieved_ids & relevant_ids) / len(relevant_ids)


def context_precision(
    retrieved_chunk_ids: list[str],
    relevant_chunk_ids: list[str],
    scores: list[float],
) -> float:
    """Return average precision over retrieved chunks ranked by score."""
    if len(retrieved_chunk_ids) != len(scores):
        raise ValueError("retrieved_chunk_ids and scores must have the same length")

    relevant_ids = set(relevant_chunk_ids)
    if not relevant_ids:
        return 0.0

    ranked_chunks = sorted(
        enumerate(zip(retrieved_chunk_ids, scores, strict=True)),
        key=lambda item: (-item[1][1], item[0]),
    )

    precision_sum = 0.0
    relevant_seen = 0
    matched_relevant_ids: set[str] = set()

    for rank, (_, (chunk_id, _score)) in enumerate(ranked_chunks, start=1):
        if chunk_id not in relevant_ids or chunk_id in matched_relevant_ids:
            continue

        matched_relevant_ids.add(chunk_id)
        relevant_seen += 1
        precision_sum += relevant_seen / rank

    return precision_sum / len(relevant_ids)
