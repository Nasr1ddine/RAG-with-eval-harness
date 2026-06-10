from __future__ import annotations

import re
from typing import Final

from openai import AsyncOpenAI

_FLOAT_RE: Final = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)")


async def faithfulness(answer: str, context: str, llm_model: str = "gpt-4o-mini") -> float:
    """Judge whether an answer contains only claims supported by the context."""
    prompt = (
        "Rate 0.0-1.0: does the answer contain ONLY claims supported by "
        "the context? 1.0=fully grounded, 0.0=hallucinated. Return only a float.\n\n"
        f"Context:\n{context}\n\n"
        f"Answer:\n{answer}"
    )
    return await _judge_score(prompt, llm_model)


async def answer_relevance(
    query: str,
    answer: str,
    llm_model: str = "gpt-4o-mini",
) -> float:
    """Judge how completely and directly an answer addresses a query."""
    prompt = (
        "Rate 0.0-1.0: how completely and directly does the answer address "
        "the query? 1.0=complete, 0.0=irrelevant. Return only a float.\n\n"
        f"Query:\n{query}\n\n"
        f"Answer:\n{answer}"
    )
    return await _judge_score(prompt, llm_model)


async def _judge_score(prompt: str, llm_model: str) -> float:
    client = AsyncOpenAI()
    response = await client.chat.completions.create(
        model=llm_model,
        messages=[
            {
                "role": "system",
                "content": "You are a strict evaluation judge. Return only a numeric score.",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
        max_tokens=10,
    )

    content = response.choices[0].message.content
    if content is None:
        raise ValueError("Judge response did not contain content")
    return _parse_score(content)


def _parse_score(raw_score: str) -> float:
    match = _FLOAT_RE.search(raw_score.strip())
    if match is None:
        raise ValueError("Judge response did not contain a float")
    return _clamp(float(match.group(0)))


def _clamp(score: float) -> float:
    return max(0.0, min(1.0, score))
