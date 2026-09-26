"""search CLI: LLM search + crawl, write citations.jsonl + sources.jsonl.

API keys are loaded from ``.env``; if a key is missing the call
simply fails.

Typical 1-query smoke test:

    python -m src.search.run \
        --input  samples/queries.jsonl \
        --models claude-haiku-4-5-20251001 \
        --output samples/_out
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse, urlunparse, parse_qsl, urlencode

from ..lib import config as cfg
from ..lib import io as _io
from ..lib import log as _log
from ..lib import models as _m
from ..lib.llm import get_search_fn
from . import extractors

LOG = _log.setup_logger("search")


# ───────────────────────────── URL normalization ───────────────────────────

_TRACK = ("utm_", "fbclid", "gclid", "msclkid", "mc_eid", "mc_cid", "_hsenc", "_hsmi")


def normalise_url(url: str) -> str:
    p = urlparse(url)
    qs = [(k, v) for k, v in parse_qsl(p.query, keep_blank_values=True)
          if not any(k.startswith(t) for t in _TRACK)]
    return urlunparse(p._replace(query=urlencode(qs), fragment=""))


def url_id_for(url: str) -> str:
    """Stable ``S0123456`` style id derived from sha1 of normalised URL."""
    h = hashlib.sha1(url.encode("utf-8")).hexdigest()
    n = int(h[:14], 16) % 9_999_999 + 1
    return f"S{n:07d}"


# ─────────────────────────── crawl-state policy ────────────────────────────


def crawl_state(status: str, content: str | None) -> tuple[str, str | None, int]:
    """Return ``(crawl_yn, failure_category, clen)`` matching paper-time policy.

    Only non-empty successful fetches are coded ``crawl_yn='Y'``; empty
    bodies and outright failures are coded ``N`` so SP/SD/ST remain NULL.
    The released ``sources.parquet`` ships only ``Y`` rows, so the failure
    strings below appear only on re-runs that hit crawl failures.
    """
    body_len = len(content) if content else 0
    if status == "ok" and body_len > 0:
        return "Y", None, body_len
    if status == "ok" and body_len == 0:
        return "N", "empty body (content length 0)", 0
    return "N", "crawl failed", 0


# ─────────────────────────────── retry wrapper ─────────────────────────────

async def with_retry(coro_fn, max_retries: int = 3, base_delay: float = 2.0):
    last: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            return await coro_fn()
        except Exception as exc:
            last = exc
            if attempt == max_retries:
                break
            es = str(exc).lower()
            delay = 60.0 if ("rate limit" in es or "429" in es) else base_delay * (2 ** (attempt - 1))
            LOG.warning("retry %d (%s) in %.0fs", attempt, exc, delay)
            await asyncio.sleep(delay)
    raise last  # type: ignore[misc]


# ────────────────────────────────── pipeline ───────────────────────────────


async def _llm_call_one(query: dict, model_id: str) -> tuple[str | None, dict | None]:
    """LLM phase only — safe to run concurrently across (query, model) pairs."""
    qid       = query["query_id"]
    qtitle    = query["question_title"]
    api_key   = cfg.api_key_for(model_id)
    search_fn = get_search_fn(model_id)

    LOG.info("search %s x %s ...", qid, model_id)
    t0 = time.time()
    try:
        raw, meta = await with_retry(
            lambda: search_fn(qtitle, model_id, api_key, cfg.SYSTEM_PROMPT)
        )
    except Exception as exc:
        LOG.error("search failed %s x %s: %s", qid, model_id, exc)
        return None, None
    LOG.info("  %s x %s -> response %d chars in %.1fs",
             qid, model_id, len(raw or ""), time.time() - t0)
    return raw, meta


async def process_one(
    query: dict,
    model_id: str,
    browser_ctx,
    *,
    cit_id_seq: list[int],
    sources_seen: dict[str, dict],
    raw_to_canonical: dict[str, str],
    citations_out: list[dict],
    sources_out: list[dict],
    contents_out: list[dict] | None = None,
    raw_responses_out: list[dict] | None = None,
    raw: str | None = None,
    meta: dict | None = None,
) -> None:
    qid       = query["query_id"]
    provider  = cfg.MODEL_REGISTRY[model_id]
    short     = _m.MODEL_SHORT[model_id]

    if raw is None:
        raw, meta = await _llm_call_one(query, model_id)
    if raw is None:
        return

    if raw_responses_out is not None:
        raw_responses_out.append({
            "query_id":     qid,
            "model_short":  short,
            "raw_response": raw or "",
        })

    citations = extractors.extract(provider, raw, meta or {})
    LOG.info("  -> %d citations extracted (%s x %s)", len(citations), qid, model_id)

    # crawl unique RAW URLs (LLM-emitted, before redirect resolution)
    # raw_to_canonical maps raw LLM URL -> canonical (normalised resolved URL)
    # so citations emitted later can find the post-redirect url_id.
    to_crawl = []
    seen_raw: set[str] = set()
    for c in citations:
        u_raw = normalise_url(c.url)
        if u_raw not in seen_raw and u_raw not in raw_to_canonical:
            seen_raw.add(u_raw)
            to_crawl.append(u_raw)
    if to_crawl:
        from ..lib import crawler as crawl  # lazy import; playwright optional
        LOG.info("  -> crawling %d new URLs", len(to_crawl))
        url_pairs = list(enumerate(to_crawl, start=1))
        crawl_results = await crawl.crawl_all(
            browser_ctx, url_pairs,
            concurrency=cfg.CRAWL_CONCURRENCY,
            timeout_ms=cfg.CRAWL_TIMEOUT_MS,
            max_chars=cfg.MAX_CONTENT_CHARS,
        )
        for _, raw_url, content, status, ts, final_url in crawl_results:
            # canonical = post-redirect URL with UTM stripped; on crawl
            # failure, fall back to the raw URL so the citation can still
            # be tracked (matches pre-refactor behaviour).
            canonical = normalise_url(final_url) if status == "ok" else raw_url
            raw_to_canonical[raw_url] = canonical
            if canonical in sources_seen:
                continue
            url_id = url_id_for(canonical)
            crawl_yn, failure, clen = crawl_state(status, content)
            sources_seen[canonical] = {
                "url_id":           url_id,
                "source_url":       canonical,
                "host_tier":        _classify_host(canonical),
                "crawl_yn":         crawl_yn,
                "failure_category": failure,
                "clen":             clen,
                "crawled_at_first": ts.replace(tzinfo=None).isoformat()
                                     if isinstance(ts, datetime) else ts,
            }
            sources_out.append(sources_seen[canonical])
            if contents_out is not None and crawl_yn == "Y":
                contents_out.append({
                    "url_id":         url_id,
                    "source_url":     canonical,
                    "source_content": content,
                })

    # emit citation rows (use canonical url_id, not the raw LLM URL)
    for c in citations:
        u_raw = normalise_url(c.url)
        canonical = raw_to_canonical.get(u_raw, u_raw)
        sid = sources_seen.get(canonical, {}).get("url_id") or url_id_for(canonical)
        cit_id_seq[0] += 1
        row = {
            "cit_id":          cit_id_seq[0],
            "query_id":        qid,
            "url_id":          sid,
            "model":           model_id,
            "model_short":     short,
            "provider":        _m.provider_of(short),
            "cited_sentence":  c.cited_sentence,
            "cited_len":       len(c.cited_sentence),
            "ASF_label":        None,         # filled by judge
            "evaluation_yn":   "Y" if (sources_seen.get(canonical, {}).get("crawl_yn") == "Y"
                                       and len(c.cited_sentence) >= 20
                                       and len(c.cited_sentence.split()) >= 5) else "N",
            "evaluation_fail_reason": _eval_fail_reason(sources_seen.get(canonical, {}), c.cited_sentence),
        }
        citations_out.append(row)


def _eval_fail_reason(src: dict, cited: str) -> str | None:
    if src.get("crawl_yn") != "Y":
        return "crawl_fail"
    if len(cited) < 20:
        return "cited_too_short"
    if len(cited.split()) < 5:
        return "cited_too_few_words"
    return None


# Minimal regex-based host classifier (paper §B.6.4 spec; full 49-rule
# table lives in lib/host_tier.py and can be imported once finalised).
_HOST_RULES: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\.(?:edu|ac\.[a-z]{2,3})$",         re.I), "Academic"),
    (re.compile(r"^(?:www\.)?(?:arxiv|nature|science|nih|pubmed|sciencedirect|springer|wiley|frontiersin)\.",
                                                    re.I), "Academic"),
    (re.compile(r"\.gov(?:\.|$)|\.org$|\.int$",      re.I), "Gov/Org"),
    (re.compile(r"(?:wikipedia|stackexchange|stackoverflow|reddit|quora|discourse)\.",
                                                    re.I), "Forum/Q&A"),
    (re.compile(r"(?:medium|substack|wordpress|blogspot|tumblr|github\.io|twitter|x\.com|facebook|linkedin|instagram|tiktok|youtube)\.",
                                                    re.I), "Social/Blog"),
    (re.compile(r"(?:nytimes|bbc|cnn|reuters|theguardian|washingtonpost|bloomberg|wsj|economist|forbes|techcrunch)\.",
                                                    re.I), "News"),
]


def _classify_host(url: str) -> str:
    host = (urlparse(url).hostname or "").lower()
    for rx, tier in _HOST_RULES:
        if rx.search(host):
            return tier
    return "Commercial/Other"


# ────────────────────────────────── main ───────────────────────────────────


async def amain(args) -> None:
    # queries source: parquet or jsonl (schema-validated either way)
    in_path = Path(args.input)
    if in_path.suffix == ".parquet":
        import pandas as pd
        df = pd.read_parquet(in_path)
        queries = df.to_dict(orient="records")
        for i, row in enumerate(queries):
            _io.validate_row("queries", row, idx=i)
    else:
        queries = list(_io.read_jsonl(in_path, schema="queries"))
    LOG.info("loaded %d queries; %d models", len(queries), len(args.models))

    out_dir: Path = args.output
    out_dir.mkdir(parents=True, exist_ok=True)

    cit_id_seq        = [args.cit_id_start - 1]
    sources_seen:     dict[str, dict] = {}
    raw_to_canonical: dict[str, str] = {}
    citations_out:    list[dict] = []
    sources_out:      list[dict] = []
    contents_out:     list[dict] | None = [] if args.save_content else None
    raw_responses_out: list[dict] = []

    from playwright.async_api import async_playwright  # lazy import
    from ..lib import crawler as crawl
    async with async_playwright() as pw:
        browser, ctx = await crawl.create_browser(pw)
        try:
            for q in queries:
                # Phase 1: parallel LLM calls across all models for this query.
                raw_metas = await asyncio.gather(
                    *[_llm_call_one(q, m) for m in args.models],
                    return_exceptions=False,
                )
                # Phase 2: serial post-process (deterministic cit_id; shared
                # state mutations stay race-free).
                for m, (raw, meta) in zip(args.models, raw_metas):
                    if raw is None:
                        continue
                    await process_one(
                        q, m, ctx,
                        cit_id_seq=cit_id_seq,
                        sources_seen=sources_seen,
                        raw_to_canonical=raw_to_canonical,
                        citations_out=citations_out,
                        sources_out=sources_out,
                        contents_out=contents_out,
                        raw_responses_out=raw_responses_out,
                        raw=raw, meta=meta,
                    )
        finally:
            await crawl.close_browser(browser)

    n_cit = _io.write_jsonl(out_dir / "citations.jsonl", citations_out, schema="citations")
    n_src = _io.write_jsonl(out_dir / "sources.jsonl",   sources_out,   schema="sources")
    n_resp = _io.write_jsonl(out_dir / "model_responses.jsonl",
                              raw_responses_out, schema="model_responses")
    msg = f"search summary: {n_cit:,} citations, {n_src:,} sources, {n_resp:,} responses"
    if contents_out is not None:
        n_con = _io.write_jsonl(out_dir / "contents.jsonl", contents_out)
        msg += f", {n_con:,} contents"
        LOG.info("wrote contents=%d -> %s", n_con, out_dir / "contents.jsonl")
    LOG.info("wrote citations=%d sources=%d responses=%d -> %s",
             n_cit, n_src, n_resp, out_dir)
    print(f"\n{msg} -> {out_dir}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input",  required=True, type=Path,
                     help="queries.jsonl from s01_query")
    ap.add_argument("--models", nargs="+", required=True,
                     help="full model ids (see CiteTrace.src.lib.config.MODEL_REGISTRY)")
    ap.add_argument("--output", required=True, type=Path,
                     help="output dir for citations.jsonl + sources.jsonl")
    ap.add_argument("--cit-id-start", type=int, default=1,
                     help="starting cit_id (use a high offset for incremental runs)")
    ap.add_argument("--save-content", action="store_true",
                     help="also write contents.jsonl ({url_id, source_url, source_content}); "
                          "needed for offline judge.")
    args = ap.parse_args()

    for m in args.models:
        if m not in cfg.MODEL_REGISTRY:
            ap.error(f"unknown model_id: {m!r}; see CiteTrace.src.lib.config.MODEL_REGISTRY")
        if not cfg.has_api_key(m):
            ap.error(f"no API key for {m} (env var {cfg.API_KEY_ENV[cfg.MODEL_REGISTRY[m]]})")

    asyncio.run(amain(args))


if __name__ == "__main__":
    main()
