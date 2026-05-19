"""JSONL pipeline outputs → HF-canonical parquet 4-tuple.

Bridges the gap between the schema-validated JSONL emitted by
``src/search/run.py`` + ``src/judge/run.py`` and the column shape that
``hf_canonical/data/*.parquet`` ships with.

HF parquet column lists this layer enforces:

* **queries.parquet**          ``query_id, site, category, question_title,
                                 question_url, QI_label``
* **sources.parquet**          ``url_id, source_url, host_tier, crawl_yn,
                                 clen, SP_label, SD_label, ST_label,
                                 crawled_at_first``
* **citations.parquet**        ``cit_id, query_id, url_id, model_short,
                                 provider, cited_sentence, cited_len,
                                 ASF_label`` — only ``evaluation_yn='Y'``.
* **model_responses.parquet**  ``query_id, model_short, raw_response``
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..lib import io as _io


HF_QUERIES_COLS   = ["query_id", "site", "category", "question_title",
                     "question_url", "QI_label"]
HF_SOURCES_COLS   = ["url_id", "source_url", "host_tier", "crawl_yn", "clen",
                     "SP_label", "SD_label", "ST_label", "crawled_at_first"]
HF_CITATIONS_COLS = ["cit_id", "query_id", "url_id", "model_short", "provider",
                     "cited_sentence", "cited_len", "ASF_label"]
HF_RESPONSES_COLS = ["query_id", "model_short", "raw_response"]


def _project(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Reorder + add missing columns as NA, drop everything else."""
    for c in cols:
        if c not in df.columns:
            df[c] = pd.NA
    return df[cols].copy()


def to_hf_parquets(jsonl_dir: Path, out_dir: Path) -> dict[str, Path]:
    """Read pipeline JSONL outputs from ``jsonl_dir`` and emit HF parquets.

    Expected layout under ``jsonl_dir``:
        {search,judge}/queries.judged.jsonl  OR  queries.judged.jsonl
        {search,judge}/sources.judged.jsonl  OR  sources.judged.jsonl
        {search,judge}/citations.judged.jsonl OR  citations.judged.jsonl
        search/model_responses.jsonl          OR  model_responses.jsonl

    Resolves both flat and nested layouts (the pipeline.sh layout writes
    judge/ outputs, search/ outputs side by side under one parent dir).
    """
    jsonl_dir = Path(jsonl_dir)
    out_dir   = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def _find(name: str) -> Path:
        for cand in (jsonl_dir / name,
                     jsonl_dir / "judge" / name,
                     jsonl_dir / "search" / name):
            if cand.exists():
                return cand
        raise FileNotFoundError(f"could not find {name} under {jsonl_dir}")

    # queries — final QI labels come from judge stage
    q_rows = _io.read_jsonl_list(_find("queries.judged.jsonl"), schema="queries")
    q = _project(pd.DataFrame(q_rows), HF_QUERIES_COLS)

    # sources — drop pipeline-state columns (failure_category, etc.)
    s_rows = _io.read_jsonl_list(_find("sources.judged.jsonl"), schema="sources")
    s = _project(pd.DataFrame(s_rows), HF_SOURCES_COLS)

    # citations — keep only evaluable rows, then drop pipeline-state cols
    c_rows = _io.read_jsonl_list(_find("citations.judged.jsonl"), schema="citations")
    c_df = pd.DataFrame(c_rows)
    if "evaluation_yn" in c_df.columns:
        c_df = c_df[c_df["evaluation_yn"] == "Y"].copy()
    c = _project(c_df, HF_CITATIONS_COLS)

    # model_responses — written by search stage (no judge step)
    r_rows = _io.read_jsonl_list(_find("model_responses.jsonl"),
                                  schema="model_responses")
    r = _project(pd.DataFrame(r_rows), HF_RESPONSES_COLS)

    paths = {
        "queries.parquet":         out_dir / "queries.parquet",
        "sources.parquet":         out_dir / "sources.parquet",
        "citations.parquet":       out_dir / "citations.parquet",
        "model_responses.parquet": out_dir / "model_responses.parquet",
    }
    q.to_parquet(paths["queries.parquet"],         index=False)
    s.to_parquet(paths["sources.parquet"],         index=False)
    c.to_parquet(paths["citations.parquet"],       index=False)
    r.to_parquet(paths["model_responses.parquet"], index=False)
    return paths


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--jsonl-dir", required=True, type=Path,
                    help="Directory holding (or containing 'search'/'judge' subdirs of) "
                         "the pipeline JSONL outputs")
    ap.add_argument("--out-dir",   required=True, type=Path,
                    help="Directory to write the four HF-format parquet files")
    args = ap.parse_args()
    paths = to_hf_parquets(args.jsonl_dir, args.out_dir)
    for name, p in paths.items():
        print(f"  wrote {name}  ({p.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
