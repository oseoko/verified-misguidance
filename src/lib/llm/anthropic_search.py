"""
Anthropic Messages API search module.
Models: claude-sonnet-4-6, claude-haiku-4-5-20251001
Tool: web_search_20250305

Returns (raw_response: str, metadata: dict).
"""
from anthropic import AsyncAnthropic


async def search(
    query: str,
    model_id: str,
    api_key: str,
    system_prompt: str,
) -> tuple[str, dict]:
    client = AsyncAnthropic(api_key=api_key)

    response = await client.messages.create(
        model=model_id,
        max_tokens=8192,
        temperature=0,
        system=system_prompt,
        tools=[{"type": "web_search_20250305", "name": "web_search"}],
        messages=[{"role": "user", "content": f"{query}"}],
        extra_headers={"anthropic-beta": "web-search-2025-03-05"},
    )

    # ── Extract text + native citations with positional tracking ────────────
    text_parts: list[str] = []
    text_block_count = 0
    native_citations: list[dict] = []
    cumulative_offset = 0

    for block in (response.content or []):
        btype = getattr(block, "type", "")

        if btype == "text":
            block_text = getattr(block, "text", "")
            block_start = cumulative_offset
            block_end = block_start + len(block_text)

            text_parts.append(block_text)
            text_block_count += 1
            cumulative_offset = block_end + 1  # +1 for "\n" joiner

            for cit in (getattr(block, "citations", None) or []):
                native_citations.append({
                    "cited_text":  getattr(cit, "cited_text", ""),
                    "url":         getattr(cit, "url", ""),
                    "title":       getattr(cit, "title", ""),
                    "start_index": block_start,
                    "end_index":   block_end,
                })

    raw_response = "\n".join(text_parts).strip()

    # ── Extract metadata ──────────────────────────────────────────────────────
    search_queries: list[dict] = []
    candidate_urls: list[list[dict]] = []

    for block in (response.content or []):
        btype = getattr(block, "type", "")

        is_search_call = (
            btype == "server_tool_use"
            or (btype == "tool_use" and getattr(block, "name", "") == "web_search")
        )
        if is_search_call:
            inp = getattr(block, "input", {}) or {}
            search_queries.append({"query": inp.get("query", "")})
            continue

        is_search_result = btype in ("web_search_tool_result", "tool_result")
        if is_search_result:
            content = getattr(block, "content", None) or []
            urls_in_result: list[dict] = []
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict):
                        urls_in_result.append({
                            "url":      item.get("url", ""),
                            "title":    item.get("title", ""),
                            "page_age": item.get("page_age", ""),
                        })
                    else:
                        url      = getattr(item, "url", "")
                        title    = getattr(item, "title", "")
                        page_age = getattr(item, "page_age", "")
                        if url:
                            urls_in_result.append({"url": url, "title": title, "page_age": page_age})
            if urls_in_result:
                candidate_urls.append(urls_in_result)

    usage = response.usage or {}
    metadata = {
        "input_tokens":            getattr(usage, "input_tokens", None),
        "output_tokens":           getattr(usage, "output_tokens", None),
        "cache_read_input_tokens": getattr(usage, "cache_read_input_tokens", None),
        "stop_reason":             response.stop_reason,
        "text_block_count":        text_block_count,
        "web_search_requests":     len(search_queries),
        "search_queries":          search_queries,
        "candidate_urls":          candidate_urls,
        "native_citations":        native_citations,
    }

    return raw_response, metadata
