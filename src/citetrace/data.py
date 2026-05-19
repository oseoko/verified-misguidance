"""CiteTrace dataset access utilities.

The recommended path for paper-table reproduction is to load the
precomputed analysis_master.parquet directly. This module provides
utilities for users who want to start from the raw tables and reconstruct
analysis_master themselves (e.g. to verify the join, apply a different
filter, or use a custom scoring matrix).

Example:
    >>> import pandas as pd
    >>> from citetrace.data import build_master
    >>> queries   = pd.read_parquet("hf://datasets/oseoko/citetrace-vm/queries.parquet")
    >>> sources   = pd.read_parquet("hf://datasets/oseoko/citetrace-vm/sources.parquet")
    >>> citations = pd.read_parquet("hf://datasets/oseoko/citetrace-vm/citations.parquet")
    >>> master = build_master(queries, sources, citations)
"""
from __future__ import annotations

import pandas as pd

from .scoring import ipam_score, ssm_score, asf_score


# Snapshot of src/lib/models.py:PROVIDER. Kept inline so citetrace/ has no
# internal dependency on lib/. Casing matches the analysis_master schema
# enum (OpenAI, Anthropic, Google, xAI, Perplexity).
_MODEL_TO_PROVIDER: dict[str, str] = {
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


def _provider_of(model_short: str) -> str:
    return _MODEL_TO_PROVIDER.get(model_short, "unknown")


def build_master(
    queries: pd.DataFrame,
    sources: pd.DataFrame,
    citations: pd.DataFrame,
) -> pd.DataFrame:
    """Reconstruct analysis_master from the three raw CiteTrace tables.

    Reproduces the SQL JOIN used by the authors:
        analysis_master = citations
                          JOIN queries ON query_id
                          JOIN sources ON url_id
        + ipam_score, ssm_score, asf_score (from scoring matrices)
        + provider (from model_short lookup)

    The HF-released citations.parquet already contains only the evaluable
    pool (n=761,495, evaluation_yn='Y'); pipeline-state columns
    (evaluation_yn, evaluation_fail_reason) are omitted. If you are
    running your own search pipeline and have raw citations with
    evaluation_yn='N' rows, filter them before calling this function.

    Args:
        queries: DataFrame with columns from queries.parquet.
                 Must contain query_id, site, category, QI_label.
        sources: DataFrame with columns from sources.parquet.
                 Must contain url_id, source_url, clen, crawl_yn,
                 SP_label, SD_label, ST_label.
        citations: DataFrame with columns from citations.parquet.
                   Must contain cit_id, query_id, url_id, model_short,
                   ASF_label, cited_sentence.

    Returns:
        DataFrame matching analysis_master.parquet column ordering.
    """
    cit = citations.copy()
    if "evaluation_yn" in cit.columns:
        cit = cit[cit["evaluation_yn"] == "Y"].copy()
    src = sources.copy()
    if "crawl_yn" in src.columns:
        src = src[src["crawl_yn"] == "Y"].copy()

    df = (cit
          .merge(queries[["query_id", "site", "category", "QI_label"]],
                 on="query_id", how="inner")
          .merge(src[["url_id", "source_url", "clen",
                      "SP_label", "SD_label", "ST_label"]],
                 on="url_id", how="inner"))

    df["provider"]   = df["model_short"].map(_provider_of)
    df["asf_score"]   = df["ASF_label"].map(asf_score)
    df["ipam_score"] = df.apply(
        lambda r: ipam_score(r["QI_label"], r["SP_label"]), axis=1
    )
    df["ssm_score"] = df.apply(
        lambda r: ssm_score(r["SD_label"], r["ST_label"]), axis=1
    )
    df["crawl_yn"] = "Y"

    schema_cols = [
        "cit_id", "query_id", "site", "category", "model_short", "provider",
        "url_id", "source_url", "cited_sentence", "clen", "cited_len",
        "QI_label", "SP_label", "ASF_label", "SD_label", "ST_label",
        "ipam_score", "asf_score", "ssm_score", "crawl_yn",
    ]
    available = [c for c in schema_cols if c in df.columns]
    return df[available].reset_index(drop=True)
