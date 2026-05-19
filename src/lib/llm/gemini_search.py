"""
Google Gemini search module via google-genai SDK.
Models: gemini-3-flash-preview, gemini-3.1-pro-preview
Grounding: google_search tool

Returns (raw_response: str, metadata: dict).
"""
import logging

import google.genai as genai
from google.genai import types

logger = logging.getLogger(__name__)


async def search(
    query: str,
    model_id: str,
    api_key: str,
    system_prompt: str,
) -> tuple[str, dict]:
    client = genai.Client(api_key=api_key)

    response = await client.aio.models.generate_content(
        model=model_id,
        contents=query,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            tools=[types.Tool(google_search=types.GoogleSearch())],
            temperature=0,
            max_output_tokens=8192,
        ),
    )

    text = (response.text or "").strip()
    if not text:
        raise ValueError("gemini returned empty response")

    # ── Extract metadata ──────────────────────────────────────────────────────
    candidate = (response.candidates or [None])[0]

    usage = response.usage_metadata or {}
    input_tokens  = getattr(usage, "prompt_token_count", None)
    output_tokens = getattr(usage, "candidates_token_count", None)
    thoughts_tok  = getattr(usage, "thoughts_token_count", None)
    total_token_count = getattr(usage, "total_token_count", None)

    finish_reason = None
    if candidate:
        fr = getattr(candidate, "finish_reason", None)
        finish_reason = fr.name if hasattr(fr, "name") else str(fr) if fr else None

    model_version = getattr(response, "model_version", None)

    grounding_meta = None
    if candidate:
        grounding_meta = getattr(candidate, "grounding_metadata", None)

    web_search_queries: list[str] = []
    grounding_chunks: list[dict]  = []
    grounding_supports: list[dict] = []

    if grounding_meta:
        wsq = getattr(grounding_meta, "web_search_queries", None) or []
        web_search_queries = list(wsq)

        chunks = getattr(grounding_meta, "grounding_chunks", None) or []
        for chunk in chunks:
            web = getattr(chunk, "web", None)
            if web:
                grounding_chunks.append({
                    "title": getattr(web, "title", ""),
                    "uri":   getattr(web, "uri", ""),
                })

        supports = getattr(grounding_meta, "grounding_supports", None) or []
        for sup in supports:
            seg = getattr(sup, "segment", None)
            text_seg = getattr(seg, "text", "") if seg else ""
            indices = list(getattr(sup, "grounding_chunk_indices", None) or [])
            grounding_supports.append({
                "chunk_indices": indices,
                "text":          text_seg,
            })

    metadata = {
        "input_tokens":         input_tokens,
        "output_tokens":        output_tokens,
        "total_token_count":    total_token_count,
        "thoughts_token_count": thoughts_tok,
        "finish_reason":        finish_reason,
        "model_version":        model_version,
        "web_search_queries":   web_search_queries,
        "grounding_chunks":     grounding_chunks,
        "grounding_supports":   grounding_supports,
    }

    return text, metadata
