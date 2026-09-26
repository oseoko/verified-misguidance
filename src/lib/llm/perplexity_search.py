"""Perplexity Sonar search via OpenAI-compatible Chat Completions API."""
from __future__ import annotations

from openai import AsyncOpenAI


async def search(
    query: str,
    model_id: str,
    api_key: str,
    system_prompt: str,
) -> tuple[str, dict]:
    client = AsyncOpenAI(
        base_url="https://api.perplexity.ai",
        api_key=api_key,
    )

    response = await client.chat.completions.create(
        model=model_id,
        temperature=0,
        max_tokens=8192,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": query},
        ],
    )

    raw_response = (response.choices[0].message.content or "").strip()
    if not raw_response:
        raise ValueError("perplexity returned empty response")

    citations: list[str] = list(getattr(response, "citations", None) or [])
    citations_map: dict[int, str] = {i + 1: u for i, u in enumerate(citations)}

    search_results: list[dict] = []
    for sr in (getattr(response, "search_results", None) or []):
        if isinstance(sr, dict):
            search_results.append(sr)
        else:
            search_results.append({
                "title":   getattr(sr, "title", ""),
                "url":     getattr(sr, "url", ""),
                "date":    getattr(sr, "date", ""),
                "snippet": getattr(sr, "snippet", ""),
            })

    usage = getattr(response, "usage", None)
    input_tokens  = getattr(usage, "prompt_tokens", None) if usage else None
    output_tokens = getattr(usage, "completion_tokens", None) if usage else None
    total_tokens  = getattr(usage, "total_tokens", None) if usage else None

    cost: dict | None = None
    raw_cost = getattr(usage, "cost", None) if usage else None
    if raw_cost:
        if isinstance(raw_cost, dict):
            cost = raw_cost
        else:
            cost = {
                "input_tokens_cost":  getattr(raw_cost, "input_tokens_cost", None),
                "output_tokens_cost": getattr(raw_cost, "output_tokens_cost", None),
                "request_cost":       getattr(raw_cost, "request_cost", None),
                "total_cost":         getattr(raw_cost, "total_cost", None),
            }

    metadata = {
        "input_tokens":        input_tokens,
        "output_tokens":       output_tokens,
        "total_tokens":        total_tokens,
        "cost":                cost,
        "search_context_size": getattr(usage, "search_context_size", None) if usage else None,
        "finish_reason":       response.choices[0].finish_reason if response.choices else None,
        "created":             getattr(response, "created", None),
        "model":               getattr(response, "model", model_id),
        "citations":           citations_map,
        "citations_count":     len(citations),
        "search_results":      search_results,
    }

    return raw_response, metadata
