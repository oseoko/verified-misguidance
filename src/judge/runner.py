"""Judge runner: per-axis async classification with retry + bounded concurrency."""
from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import dataclass
from typing import Any

from . import prompts as P

LOG = logging.getLogger("judge.runner")


@dataclass
class JudgeResult:
    label: str | None        # axis label (e.g. "QI3", "ASF5", ...) or None on fail
    reasoning: str | None
    raw: dict[str, Any] | None
    error: str | None = None


class JudgeRunner:
    """One async OpenAI client + bounded concurrency, reused across axes."""

    def __init__(
        self,
        api_key: str,
        *,
        model: str = P.JUDGE_MODEL,
        concurrency: int = 20,
        max_retries: int = 3,
        max_source_chars: int = 30_000,
    ) -> None:
        from openai import AsyncOpenAI  # lazy: SDK import deferred to first call
        self.client    = AsyncOpenAI(api_key=api_key)
        self.model     = model
        self._sem      = asyncio.Semaphore(concurrency)
        self.max_retries = max_retries
        self.max_source_chars = max_source_chars

    @classmethod
    def from_env(cls, **kw) -> "JudgeRunner":
        key = (os.getenv("OPENAI_API_KEY") or "").strip()
        if not key:
            raise RuntimeError("OPENAI_API_KEY missing in environment")
        return cls(api_key=key, **kw)

    # ───────────────────── low-level call ────────────────────────

    async def _call(
        self,
        system_prompt: str,
        user_msg: str,
        schema: dict,
        label_field: str,
        reasoning_field: str,
    ) -> JudgeResult:
        last_err: Exception | None = None
        async with self._sem:
            for attempt in range(1, self.max_retries + 1):
                try:
                    resp = await self.client.chat.completions.create(
                        model=self.model,
                        temperature=P.JUDGE_TEMPERATURE,
                        response_format=schema,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user",   "content": user_msg},
                        ],
                    )
                    content = (resp.choices[0].message.content or "").strip()
                    parsed  = json.loads(content)
                    return JudgeResult(
                        label=parsed.get(label_field),
                        reasoning=parsed.get(reasoning_field),
                        raw=parsed,
                    )
                except Exception as exc:  # noqa: BLE001
                    last_err = exc
                    msg = str(exc).lower()
                    delay = 60.0 if ("rate limit" in msg or "429" in msg) else 2.0 * (2 ** (attempt - 1))
                    LOG.warning("judge retry %d (%s) sleeping %.0fs", attempt, exc, delay)
                    await asyncio.sleep(delay)
        return JudgeResult(label=None, reasoning=None, raw=None, error=str(last_err))

    # ───────────────────── public per-axis ────────────────────────

    async def qi(self, query: str) -> JudgeResult:
        return await self._call(
            P.SYSTEM_PROMPT_QI, P.user_msg_qi(query),
            P.SCHEMA_QI, "intent", "intent_reasoning",
        )

    async def sp(self, source_url: str, source_content: str) -> JudgeResult:
        return await self._call(
            P.SYSTEM_PROMPT_SP,
            P.user_msg_sp(source_url, source_content[: self.max_source_chars]),
            P.SCHEMA_SP, "purpose", "purpose_reasoning",
        )

    async def sd(self, source_url: str, source_content: str) -> JudgeResult:
        return await self._call(
            P.SYSTEM_PROMPT_SD,
            P.user_msg_sd(source_url, source_content[: self.max_source_chars]),
            P.SCHEMA_SD, "domain", "domain_reasoning",
        )

    async def st(self, source_url: str, source_content: str) -> JudgeResult:
        return await self._call(
            P.SYSTEM_PROMPT_ST,
            P.user_msg_st(source_url, source_content[: self.max_source_chars]),
            P.SCHEMA_ST, "source_type", "type_reasoning",
        )

    async def asf_(self, cited_sentence: str, source_content: str) -> JudgeResult:
        return await self._call(
            P.SYSTEM_PROMPT_ASF,
            P.user_msg_asf(cited_sentence, source_content[: self.max_source_chars]),
            P.SCHEMA_ASF, "verdict", "asf_reasoning",
        )


__all__ = ["JudgeRunner", "JudgeResult"]
