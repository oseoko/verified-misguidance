"""LLM provider / model lookup tables (paper §B.4.1).

Use ``MODEL_SHORT[full_id]`` to convert a raw model id (returned by the
provider SDK) to the canonical short name used across the dataset, and
``provider_of(short)`` to look up the provider label.
"""
from __future__ import annotations

from typing import Final


# {full_model_id: short_name}
MODEL_SHORT: Final[dict[str, str]] = {
    # OpenAI
    "gpt-5-2025-08-07": "gpt-5",
    "gpt-5-mini-2025-08-07": "gpt-5-mini",
    # Anthropic
    "claude-sonnet-4-6": "claude-sonnet",
    "claude-haiku-4-5-20251001": "claude-haiku",
    # Google
    "gemini-3-flash-preview": "gemini-flash",
    "gemini-3.1-pro-preview": "gemini-pro",
    # xAI
    "grok-4-1-fast-non-reasoning": "grok-nr",
    "grok-4-1-fast-reasoning": "grok-r",
    # Perplexity
    "sonar": "sonar",
    "sonar-reasoning-pro": "sonar-rp",
}

# {short_name: provider_label}
PROVIDER: Final[dict[str, str]] = {
    "gpt-5":         "OpenAI",
    "gpt-5-mini":    "OpenAI",
    "claude-sonnet": "Anthropic",
    "claude-haiku":  "Anthropic",
    "gemini-flash":  "Google",
    "gemini-pro":    "Google",
    "grok-nr":       "xAI",
    "grok-r":        "xAI",
    "sonar":         "Perplexity",
    "sonar-rp":      "Perplexity",
}

# Reverse: {short_name: full_id}
MODEL_FULL: Final[dict[str, str]] = {v: k for k, v in MODEL_SHORT.items()}


def provider_of(model_id_or_short: str) -> str:
    """Resolve any incoming model identifier to provider label."""
    s = model_id_or_short
    if s in PROVIDER:
        return PROVIDER[s]
    if s in MODEL_SHORT:
        return PROVIDER[MODEL_SHORT[s]]
    raise KeyError(f"unknown model: {model_id_or_short!r}")


__all__ = [
    "MODEL_SHORT",
    "MODEL_FULL",
    "PROVIDER",
    "provider_of",
]
