"""
xAI Grok search module via OpenAI-compatible Responses API.
Models: grok-4-1-fast-non-reasoning, grok-4-1-fast-reasoning
Tool: web_search (server-side)

Returns (raw_response: str, metadata: dict).
Uses AsyncOpenAI with base_url pointed to xAI.
"""
from openai import AsyncOpenAI


async def search(
    query: str,
    model_id: str,
    api_key: str,
    system_prompt: str,
) -> tuple[str, dict]:
    client = AsyncOpenAI(api_key=api_key, base_url="https://api.x.ai/v1")

    response = await client.responses.create(
        model=model_id,
        tools=[{"type": "web_search"}],
        instructions=system_prompt,
        input=query,
        temperature=0,
        max_output_tokens=8192,
    )

    # ── Extract text + annotations (OpenAI-compatible Responses API) ──────────
    text_parts: list[str] = []
    annotations: list[dict] = []

    for item in (response.output or []):
        for content in (getattr(item, "content", None) or []):
            if getattr(content, "type", "") == "text":
                text_parts.append(getattr(content, "text", ""))
            # xAI exposes citations either flat (annotation.url) or nested
            # (annotation.url_citation / annotation.web_citation) depending on
            # SDK version; handle both so the extractor can slice raw_response.
            for ann in (getattr(content, "annotations", None) or []):
                url_obj = (getattr(ann, "url_citation", None)
                           or getattr(ann, "web_citation", None))
                if url_obj:
                    url   = getattr(url_obj, "url", "")
                    title = getattr(url_obj, "title", "")
                    start = getattr(url_obj, "start_index", None)
                    end   = getattr(url_obj, "end_index", None)
                else:
                    url   = getattr(ann, "url", "")
                    title = getattr(ann, "title", "")
                    start = getattr(ann, "start_index", None)
                    end   = getattr(ann, "end_index", None)
                if url:
                    annotations.append({"url": url, "title": title,
                                         "start_index": start, "end_index": end})

    raw_response = "".join(text_parts).strip()
    if not raw_response:
        raw_response = (getattr(response, "output_text", "") or "").strip()

    # ── All citations (URL list from search; broader than inline-cited set) ───
    all_citations: list[str] = []
    for cite in (getattr(response, "citations", None) or []):
        url = getattr(cite, "url", "") if not isinstance(cite, str) else cite
        if url:
            all_citations.append(url)

    # ── Extract metadata ───────────────────────────────────────────────────────
    usage = getattr(response, "usage", None)
    input_tokens  = getattr(usage, "input_tokens", None)
    output_tokens = getattr(usage, "output_tokens", None)
    total_tokens  = getattr(usage, "total_tokens", None)

    search_actions: list[dict] = []
    reasoning_item_count = 0

    for item in (response.output or []):
        itype = getattr(item, "type", "")
        if itype == "reasoning":
            reasoning_item_count += 1
        elif itype == "web_search_call":
            action = getattr(item, "action", None)
            action_type = getattr(action, "type", "search") if action else "search"
            if action_type == "open_page":
                search_actions.append({
                    "type": "open_page",
                    "url":  getattr(action, "url", ""),
                })
            else:
                qs = getattr(action, "queries", None) if action else None
                if not qs:
                    q = getattr(action, "query", "") if action else ""
                    qs = [q] if q else []
                search_actions.append({
                    "type":    "search",
                    "queries": list(qs),
                })

    metadata = {
        "input_tokens":         input_tokens,
        "output_tokens":        output_tokens,
        "total_tokens":         total_tokens,
        "finish_reason":        getattr(response, "status", None),
        "created_at":           getattr(response, "created_at", None),
        "reasoning_item_count": reasoning_item_count,
        "search_actions":       search_actions,
        "annotations":          annotations,
        "all_citations":        all_citations,
    }

    return raw_response, metadata
