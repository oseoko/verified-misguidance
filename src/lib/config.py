"""Run-time configuration for search / judge.

Model registry, system prompt, crawler settings, and API-key resolution.
Mirrors paper §B.2 / §B.4 with the minimal-intervention prompt.
"""
from __future__ import annotations

import os
from typing import Final

from dotenv import load_dotenv

from . import paths

if paths.ENV_FILE.exists():
    load_dotenv(paths.ENV_FILE, override=False)


# ── System prompt (Appendix B.2, minimal intervention) ─────────────────────

SYSTEM_PROMPT: Final[str] = (
    "You are a search-augmented LLM.\n"
    "\n"
    "Use the search tool to retrieve relevant external knowledge "
    "and provide an accurate, comprehensive answer to the following question."
)
"""Paper §B.2: identical for all 10 models. Two paragraphs separated by a
blank line. No instructions on citation format, count, or preferred
sources."""

PROMPT_VERSION: Final[str] = "minimal_v1"


# ── Model registry ────────────────────────────────────────────────────────

# {model_id: provider}
MODEL_REGISTRY: Final[dict[str, str]] = {
    "gpt-5-2025-08-07":            "openai",
    "gpt-5-mini-2025-08-07":       "openai",
    "claude-sonnet-4-6":           "anthropic",
    "claude-haiku-4-5-20251001":   "anthropic",
    "gemini-3-flash-preview":      "google",
    "gemini-3.1-pro-preview":      "google",
    "grok-4-1-fast-non-reasoning": "xai",
    "grok-4-1-fast-reasoning":     "xai",
    "sonar":                       "perplexity",
    "sonar-reasoning-pro":         "perplexity",
}

API_KEY_ENV: Final[dict[str, str]] = {
    "openai":     "OPENAI_API_KEY",
    "anthropic":  "ANTHROPIC_API_KEY",
    "google":     "GOOGLE_API_KEY",
    "xai":        "XAI_API_KEY",
    "perplexity": "PERPLEXITY_API_KEY",
}


def api_key_for(model_id: str) -> str:
    provider = MODEL_REGISTRY[model_id]
    env = API_KEY_ENV[provider]
    key = os.getenv(env, "").strip()
    if not key:
        raise RuntimeError(
            f"missing env var {env} for provider {provider}; "
            f"set it in {paths.ENV_FILE} or the shell."
        )
    return key


def has_api_key(model_id: str) -> bool:
    """Best-effort check without raising."""
    try:
        api_key_for(model_id)
        return True
    except RuntimeError:
        return False


# ── Crawler (paper §B.6) ──────────────────────────────────────────────────

CRAWL_TIMEOUT_MS:    Final[int] = 15_000   # 15 s page-collection timeout
CRAWL_CONCURRENCY:   Final[int] = 5        # global concurrency
MAX_CONTENT_CHARS:   Final[int] = 50_000   # content cap


__all__ = [
    "SYSTEM_PROMPT", "PROMPT_VERSION",
    "MODEL_REGISTRY", "API_KEY_ENV",
    "api_key_for", "has_api_key",
    "CRAWL_TIMEOUT_MS", "CRAWL_CONCURRENCY", "MAX_CONTENT_CHARS",
]
