"""judge CLI: classify queries.jsonl + sources.jsonl + citations.jsonl.

Source content is required for SP/SD/ST/AS classification.
Provide it via ``--content-jsonl`` (mapping ``url_id -> source_content``).

API keys are loaded from ``.env``.

Outputs (JSONL) are written to ``--output-dir``:

* ``queries.judged.jsonl``    - QI_label filled
* ``sources.judged.jsonl``    - SP_label / SD_label / ST_label filled
* ``citations.judged.jsonl``  - ASF_label filled (evaluation_yn updated)
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Iterable

from ..lib import config as _cfg  # noqa: F401  (side-effect: load_dotenv)
from ..lib import io as _io
from ..lib import log as _log
from .runner import JudgeRunner, JudgeResult

LOG = _log.setup_logger("judge")


# ─────────────────────── source-content resolution ─────────────────────────


class ContentStore:
    """Lazy fetch of crawled source content keyed by ``url_id``."""

    def __init__(self, content_jsonl: Path | None = None) -> None:
        self._cache: dict[str, str] = {}
        if content_jsonl:
            for row in _io.read_jsonl(content_jsonl):
                k = row.get("url_id") or row.get("source_url")
                if k:
                    self._cache[k] = row.get("source_content", "")

    def get(self, url_id: str, source_url: str) -> str | None:
        if url_id in self._cache:
            return self._cache[url_id]
        if source_url in self._cache:
            return self._cache[source_url]
        return None


# ────────────────────────── per-stage processors ───────────────────────────


async def judge_queries(judge: JudgeRunner, rows: list[dict]) -> list[dict]:
    out: list[dict] = []
    pending = [judge.qi(r["question_title"]) for r in rows]
    results = await asyncio.gather(*pending, return_exceptions=False)
    for r, jr in zip(rows, results):
        new = dict(r)
        if jr.label:
            new["QI_label"] = jr.label
        out.append(new)
    return out


async def judge_sources(
    judge: JudgeRunner, rows: list[dict], store: ContentStore,
) -> list[dict]:
    out: list[dict] = []
    tasks = []
    for r in rows:
        if r.get("crawl_yn") != "Y":
            tasks.append(asyncio.sleep(0, result=(None, None, None)))
            continue
        content = store.get(r["url_id"], r.get("source_url", ""))
        if not content:
            tasks.append(asyncio.sleep(0, result=(None, None, None)))
            continue
        tasks.append(_judge_source_axes(judge, r["source_url"], content))
    results = await asyncio.gather(*tasks)
    for r, (sp, sd, st) in zip(rows, results):
        new = dict(r)
        if sp: new["SP_label"] = sp
        if sd: new["SD_label"] = sd
        if st: new["ST_label"] = st
        out.append(new)
    return out


async def _judge_source_axes(judge: JudgeRunner, url: str, content: str):
    sp_t = judge.sp(url, content)
    sd_t = judge.sd(url, content)
    st_t = judge.st(url, content)
    sp, sd, st = await asyncio.gather(sp_t, sd_t, st_t)
    return sp.label, sd.label, st.label


async def judge_citations(
    judge: JudgeRunner, rows: list[dict],
    sources_by_id: dict[str, dict], store: ContentStore,
) -> list[dict]:
    tasks = []
    contexts: list[tuple[dict, dict | None, str | None]] = []
    for r in rows:
        if r.get("evaluation_yn") != "Y":
            tasks.append(asyncio.sleep(0, result=None))
            contexts.append((r, None, None))
            continue
        src = sources_by_id.get(r["url_id"])
        if not src:
            tasks.append(asyncio.sleep(0, result=None))
            contexts.append((r, None, "crawl_fail"))
            continue
        content = store.get(r["url_id"], src.get("source_url", ""))
        if not content:
            tasks.append(asyncio.sleep(0, result=None))
            contexts.append((r, src, "crawl_fail"))
            continue
        tasks.append(judge.asf_(r["cited_sentence"], content))
        contexts.append((r, src, None))

    results = await asyncio.gather(*tasks)
    out: list[dict] = []
    for (r, src, fail), jr in zip(contexts, results):
        new = dict(r)
        if jr is None:
            new["evaluation_yn"] = "N"
            new["evaluation_fail_reason"] = fail or new.get("evaluation_fail_reason")
            out.append(new)
            continue
        if jr.label:
            new["ASF_label"]      = jr.label
            new["evaluation_yn"] = "Y"
            new["evaluation_fail_reason"] = None
        else:
            new["evaluation_yn"] = "N"
            new["evaluation_fail_reason"] = "judge_unevaluable"
        out.append(new)
    return out


# ─────────────────────────────────── main ──────────────────────────────────


async def amain(args) -> None:
    out_dir: Path = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    judge = JudgeRunner.from_env(concurrency=args.concurrency)
    store = ContentStore(args.content_jsonl)

    def _read_rows(path: Path, schema: str) -> list[dict]:
        if path.suffix == ".parquet":
            import pandas as pd
            df = pd.read_parquet(path)
            rows = df.to_dict(orient="records")
            for i, r in enumerate(rows):
                _io.validate_row(schema, r, idx=i)
            return rows
        return list(_io.read_jsonl(path, schema=schema))

    if args.queries:
        rows = _read_rows(args.queries, "queries")
        LOG.info("judging QI for %d queries", len(rows))
        out  = await judge_queries(judge, rows)
        n = _io.write_jsonl(out_dir / "queries.judged.jsonl", out, schema="queries")
        LOG.info("wrote queries.judged.jsonl (%d rows)", n)

    if args.sources:
        rows = _read_rows(args.sources, "sources")
        LOG.info("judging SP/SD/ST for %d sources", len(rows))
        out  = await judge_sources(judge, rows, store)
        n = _io.write_jsonl(out_dir / "sources.judged.jsonl", out, schema="sources")
        LOG.info("wrote sources.judged.jsonl (%d rows)", n)

    if args.citations:
        cit_rows = _read_rows(args.citations, "citations")
        if not args.sources:
            raise SystemExit("--citations requires --sources for url->content lookup")
        sources_by_id = {r["url_id"]: r for r in _read_rows(args.sources, "sources")}
        LOG.info("judging AS for %d citations", len(cit_rows))
        out = await judge_citations(judge, cit_rows, sources_by_id, store)
        n = _io.write_jsonl(out_dir / "citations.judged.jsonl", out, schema="citations")
        LOG.info("wrote citations.judged.jsonl (%d rows)", n)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries",       type=Path, default=None,
                     help="queries.jsonl  (s01_query output)")
    ap.add_argument("--sources",       type=Path, default=None,
                     help="sources.jsonl  (search output)")
    ap.add_argument("--citations",     type=Path, default=None,
                     help="citations.jsonl (search output)")
    ap.add_argument("--content-jsonl", type=Path, default=None,
                     help="jsonl with {url_id, source_content} rows "
                          "(from search --save-content)")
    ap.add_argument("--output-dir",    type=Path, required=True)
    ap.add_argument("--concurrency",   type=int,  default=20)
    args = ap.parse_args()

    if not (args.queries or args.sources or args.citations):
        ap.error("at least one of --queries / --sources / --citations is required")

    asyncio.run(amain(args))


if __name__ == "__main__":
    main()
