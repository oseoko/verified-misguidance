"""Provider dispatch for search-augmented LLM calls.

Lazy-imports per provider so missing optional SDKs don't break the
whole module.
"""
from __future__ import annotations

from typing import Awaitable, Callable

from .. import config


SearchFn = Callable[[str, str, str, str], Awaitable[tuple[str, dict]]]


def get_search_fn(model_id: str) -> SearchFn:
    """Return the provider-specific async ``search(query, model_id, key, sys_prompt)`` callable."""
    provider = config.MODEL_REGISTRY[model_id]
    if provider == "openai":
        from . import openai_search
        return openai_search.search
    if provider == "anthropic":
        from . import anthropic_search
        return anthropic_search.search
    if provider == "google":
        from . import gemini_search
        return gemini_search.search
    if provider == "xai":
        from . import grok_search
        return grok_search.search
    if provider == "perplexity":
        from . import perplexity_search
        return perplexity_search.search
    raise ValueError(f"unknown provider for model_id={model_id!r}: {provider}")


__all__ = ["get_search_fn", "SearchFn"]
