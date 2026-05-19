"""Reproduce paper §D tables from ``analysis_master.parquet``.

Usage:
    python -m CiteTrace.src.analysis.reproduce \\
        --input      data/analysis_master.parquet \\
        --output-dir results/d_tables

    # Or stream from Hugging Face
    python -m CiteTrace.src.analysis.reproduce \\
        --input      hf://datasets/oseoko/citetrace-vm/data/analysis_master.parquet \\
        --output-dir results/d_tables

Output (one TSV per paper §D table):
    D.1  tab_d_pool_composition
    D.2  tab_d_score_dist, tab_d_three_axis, tab_d_ymyl_ssfr, tab_d_ymyl_st
    D.3  tab_d_safr_per_model, tab_d_source_type_by_model
    D.4  tab_d_provider_profile, tab_d_variance,
         tab_d_model_size_paired, tab_d_reasoning_paired
    D.5  tab_d_response_exposure
    D.6  tab_d_threshold, tab_d_temporal

tab_d_temporal joins ``crawled_at_first`` from a sibling
``sources.parquet`` automatically when present. tab_d_crawl_bias
(phantom + PDF rates) requires the raw 1,271,046 pre-crawl citation
pool (filtered out before the evaluable cut) and so is not derivable
from analysis_master.parquet alone. tab_d_cases (D.7) is hand-curated
in the paper appendix and not regenerated here.
"""
from __future__ import annotations

import argparse
import logging
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from scipy import stats as _stats
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

LOG = logging.getLogger("reproduce")

# --------------------------------------------------------------------------- #
# Paper canonical labels (paper §D figures and tables)
# --------------------------------------------------------------------------- #

# Score code → paper-readable label (Appendix §D.2, Table tab_d_score_dist).
ASF_LABELS = {
    5: "Supported", 4: "Amplified", 3: "Contradicted",
    2: "Misattributed", 1: "Fabricated",
}
SS_LABELS = {
    5: "Suitable", 4: "Adequate", 3: "Borderline",
    2: "Inadequate", 1: "Unsuitable",
}
IPA_LABELS = {
    5: "Structural Match", 4: "Functional Support", 3: "Partial Relevance",
    2: "Weak Fit", 1: "Structural Conflict",
}

# ST/SP/SD code → paper-readable label (paper Tables 18, 19, 20).
ST_LABELS = {
    "ST1": "Official Institution",
    "ST2": "Paper/Research",
    "ST3": "News/Magazine",
    "ST4": "Wiki/Forum",
    "ST5": "Blog/Social",
    "ST6": "Private Company",
}
SP_LABELS = {
    "SP1": "To Promote", "SP2": "To Inform",  "SP3": "To Instruct",
    "SP4": "To Report",  "SP5": "To Discuss", "SP6": "To Opine",
}
SD_LABELS = {
    "SD1":  "Medical/Health",
    "SD2":  "Legal",
    "SD3":  "Finance",
    "SD4":  "Education",
    "SD5":  "Science",
    "SD6":  "Code/Data",
    "SD7":  "Technical",
    "SD8":  "Social/Professional",
    "SD9":  "Shopping/Travel",
    "SD10": "Everyday",
}

# YMYL = high-stakes domains (paper §D.1).
YMYL_SD = {"SD1", "SD2", "SD3"}  # Medical, Legal, Finance

# Score-distribution display order (paper tab_d_score_dist row order).
ASF_DISPLAY_ORDER = [5, 4, 3, 2, 1]
SS_DISPLAY_ORDER  = [5, 4, 3, 2, 1]
IPA_DISPLAY_ORDER = [5, 4, 3, 2, 1]

# Source Type display order in tab_d_pool_composition (sorted by count desc).
ST_PAPER_ORDER = ["ST5", "ST6", "ST1", "ST4", "ST3", "ST2"]   # Blog, Company, Official, Wiki, News, Research
SP_PAPER_ORDER = ["SP2", "SP3", "SP5", "SP1", "SP4", "SP6"]   # Inform, Instruct, Discuss, Promote, Report, Opine
SD_PAPER_ORDER = ["SD7", "SD6", "SD1", "SD5", "SD10", "SD3", "SD2", "SD8", "SD9", "SD4"]
# Technical, Code/Data, Medical, Science, Everyday, Finance, Legal, Social, Shopping, Education

# --------------------------------------------------------------------------- #
# Model registry — defines display names, providers, and pair groupings
# --------------------------------------------------------------------------- #

# Display names match paper figures/tables (Appendix §B.4.1).
MODEL_DISPLAY = {
    "claude-haiku":  "claude-haiku-4-5",
    "claude-sonnet": "claude-sonnet-4-6",
    "gpt-5":         "gpt-5",
    "gpt-5-mini":    "gpt-5-mini",
    "gemini-flash":  "gemini-3-flash",
    "gemini-pro":    "gemini-3.1-pro",
    "grok-nr":       "grok-4.1-fast (NR)",
    "grok-r":        "grok-4.1-fast (R)",
    "sonar":         "sonar",
    "sonar-rp":      "sonar-reasoning-pro",
}

PROVIDER_OF = {
    "claude-haiku":  "Anthropic",  "claude-sonnet": "Anthropic",
    "gpt-5":         "OpenAI",     "gpt-5-mini":    "OpenAI",
    "gemini-flash":  "Google",     "gemini-pro":    "Google",
    "grok-nr":       "xAI",        "grok-r":        "xAI",
    "sonar":         "Perplexity", "sonar-rp":      "Perplexity",
}

PROVIDER_ORDER = ["Anthropic", "OpenAI", "Perplexity", "Google", "xAI"]

# Within-provider model-size pairs (paper §D.4, tab_d_model_size_paired).
MODEL_SIZE_PAIRS = [
    ("Anthropic", "claude-sonnet", "claude-haiku"),  # larger first
    ("OpenAI",    "gpt-5",         "gpt-5-mini"),
    ("Google",    "gemini-pro",    "gemini-flash"),
]

# Reasoning vs non-reasoning pairs (paper §D.4, tab_d_reasoning_paired).
REASONING_PAIRS = [
    ("xAI",        "grok-r",   "grok-nr"),  # reasoning first
    ("Perplexity", "sonar-rp", "sonar"),
]


# --------------------------------------------------------------------------- #
# I/O helpers
# --------------------------------------------------------------------------- #

def write_tsv(out_dir: Path, name: str, df: pd.DataFrame, fmt: str = "%.4f") -> None:
    """Write a DataFrame as TSV with a stable float format."""
    path = out_dir / f"{name}.tsv"
    df.to_csv(path, sep="\t", index=False, float_format=fmt)
    LOG.info("wrote %s (%d rows)", name, len(df))


def _share(n: int, total: int) -> float:
    return 100.0 * n / total if total else 0.0


def _eta_squared(h: float, k: int, n: int) -> float:
    """Kruskal–Wallis effect size (Tomczak & Tomczak, 2014)."""
    if n - k <= 0:
        return 0.0
    return max((h - k + 1) / (n - k), 0.0)


# --------------------------------------------------------------------------- #
# D.1 — Source Pool Composition
# --------------------------------------------------------------------------- #

def d_pool_composition(am: pd.DataFrame, out: Path) -> None:
    """Paper Table tab_d_pool_composition — combined ST + SP + SD distribution."""
    n_total = len(am)
    rows: list[dict] = []

    for code in ST_PAPER_ORDER:
        cnt = (am.ST_label == code).sum()
        rows.append({"dimension": "Source Type", "label": ST_LABELS[code],
                     "ymyl": "", "count": int(cnt), "pct": round(_share(cnt, n_total), 2)})
    for code in SP_PAPER_ORDER:
        cnt = (am.SP_label == code).sum()
        rows.append({"dimension": "Source Purpose", "label": SP_LABELS[code],
                     "ymyl": "", "count": int(cnt), "pct": round(_share(cnt, n_total), 2)})
    for code in SD_PAPER_ORDER:
        cnt = (am.SD_label == code).sum()
        rows.append({"dimension": "Source Domain", "label": SD_LABELS[code],
                     "ymyl": "Y" if code in YMYL_SD else "",
                     "count": int(cnt), "pct": round(_share(cnt, n_total), 2)})

    write_tsv(out, "tab_d_pool_composition",
              pd.DataFrame(rows, columns=["dimension","label","ymyl","count","pct"]))


# --------------------------------------------------------------------------- #
# D.2 — Aggregate Failure Rates
# --------------------------------------------------------------------------- #

def d_score_dist(am: pd.DataFrame, out: Path) -> None:
    """Paper Table tab_d_score_dist — ASF/SS/IPA score 1-5 distribution."""
    n = len(am)
    rows = []
    for dim, col, labels, order in [
        ("ASF (Fidelity)",   "asf_score",  ASF_LABELS, ASF_DISPLAY_ORDER),
        ("SS (Suitability)", "ssm_score",  SS_LABELS,  SS_DISPLAY_ORDER),
        ("IPA (Alignment)",  "ipam_score", IPA_LABELS, IPA_DISPLAY_ORDER),
    ]:
        for s in order:
            cnt = (am[col] == s).sum()
            rows.append({
                "dimension": dim,
                "label":     f"{labels[s]} ({s})",
                "count":     int(cnt),
                "pct":       round(_share(cnt, n), 1),
            })
    write_tsv(out, "tab_d_score_dist",
              pd.DataFrame(rows, columns=["dimension", "label", "count", "pct"]))


def _eta_interp(x: float) -> str:
    """Tomczak η² interpretation: <0.01 negligible, <0.06 small, <0.14 medium."""
    if x != x:
        return ""
    if x < 0.01:  return "negligible"
    if x < 0.06:  return "small"
    if x < 0.14:  return "medium"
    return "large"


def d_three_axis(am: pd.DataFrame, out: Path) -> None:
    """Paper Table tab_d_three_axis — mean (±sd), failure rate, Kruskal η²
    grouped by model and by category."""
    rows = []
    for dim, col, fr_name in [
        ("Answer-Source Fidelity",   "asf_score",  "FFR"),
        ("Source Suitability",       "ssm_score",  "SFR"),
        ("Intent-Purpose Alignment", "ipam_score", "AFR"),
    ]:
        scores = am[col].astype(int).to_numpy()
        n = len(scores)
        m, sd = float(scores.mean()), float(scores.std(ddof=0))
        rate = 100.0 * (scores <= 2).mean()

        eta_model = eta_cat = float("nan")
        if _HAS_SCIPY:
            grp_m = [g[col].astype(int).to_numpy()
                     for _, g in am.groupby("model_short", sort=False)]
            if len(grp_m) >= 2 and all(len(g) > 0 for g in grp_m):
                H_m, _ = _stats.kruskal(*grp_m)
                eta_model = _eta_squared(H_m, len(grp_m), n)
            grp_c = [g[col].astype(int).to_numpy()
                     for _, g in am.groupby("category", sort=False)]
            if len(grp_c) >= 2 and all(len(g) > 0 for g in grp_c):
                H_c, _ = _stats.kruskal(*grp_c)
                eta_cat = _eta_squared(H_c, len(grp_c), n)

        rows.append({
            "dimension":      dim,
            "mean_1_5":       f"{m:.2f} (±{sd:.2f})",
            "failure_rate":   f"{fr_name} {rate:.1f}%",
            "eta2_model":     f"{eta_model:.3f} ({_eta_interp(eta_model)})",
            "eta2_category":  f"{eta_cat:.3f} ({_eta_interp(eta_cat)})",
        })
    write_tsv(out, "tab_d_three_axis",
              pd.DataFrame(rows, columns=["dimension", "mean_1_5", "failure_rate",
                                          "eta2_model", "eta2_category"]))


def d_ymyl_ssfr(am: pd.DataFrame, out: Path) -> None:
    """Paper Table tab_d_ymyl_ssfr — YMYL vs non-YMYL Suitability Failure Rate.

    Fisher's exact OR is reported in the paper caption (not the table) and
    is not emitted here.
    """
    is_ymyl = am.SD_label.isin(YMYL_SD)
    rows = []
    for grp_name, mask in [
        ("YMYL (Medical + Legal + Finance)", is_ymyl),
        ("non-YMYL", ~is_ymyl),
    ]:
        sub = am[mask]
        n = len(sub)
        ss = sub.ssm_score.astype(int).to_numpy()
        rows.append({
            "group":   grp_name,
            "n":       n,
            "R_SS":    round(float(ss.mean()), 3) if n else float("nan"),
            "SFR_pct": round(100.0 * (ss <= 2).mean(), 1) if n else float("nan"),
        })
    write_tsv(out, "tab_d_ymyl_ssfr",
              pd.DataFrame(rows, columns=["group", "n", "R_SS", "SFR_pct"]))


def d_ymyl_st(am: pd.DataFrame, out: Path) -> None:
    """Paper Table tab_d_ymyl_st — Source Type composition by YMYL group."""
    is_ymyl = am.SD_label.isin(YMYL_SD)
    rows = []
    n_y, n_n = int(is_ymyl.sum()), int((~is_ymyl).sum())
    for code in ST_PAPER_ORDER:
        cy = int(((am.ST_label == code) & is_ymyl).sum())
        cn = int(((am.ST_label == code) & ~is_ymyl).sum())
        rows.append({
            "source_type":     ST_LABELS[code],
            "ymyl_count":      cy,  "ymyl_pct":      round(_share(cy, n_y), 2),
            "non_ymyl_count":  cn,  "non_ymyl_pct":  round(_share(cn, n_n), 2),
        })
    rows.append({
        "source_type":    "Total",
        "ymyl_count":     n_y, "ymyl_pct":     100.0,
        "non_ymyl_count": n_n, "non_ymyl_pct": 100.0,
    })
    write_tsv(out, "tab_d_ymyl_st", pd.DataFrame(rows))


# --------------------------------------------------------------------------- #
# D.3 — Fidelity-Suitability Trade-off (per-model)
# --------------------------------------------------------------------------- #

def d_safr_per_model(am: pd.DataFrame, out: Path) -> None:
    """Paper Table tab_d_safr_per_model — per-model FFR + SFR + rank-shift Δ."""
    g = (am.groupby("model_short", sort=False)
           .agg(FFR=("asf_score", lambda s: 100.0 * (s <= 2).mean()),
                SFR=("ssm_score", lambda s: 100.0 * (s <= 2).mean()))
           .reset_index())
    g["FFR_rank"] = g["FFR"].rank(method="min").astype(int)
    g["SFR_rank"] = g["SFR"].rank(method="min").astype(int)
    g["delta_rank"] = g["FFR_rank"] - g["SFR_rank"]
    g = g.sort_values("FFR_rank").reset_index(drop=True)
    g["model"] = g["model_short"].map(MODEL_DISPLAY)
    g["FFR"] = g["FFR"].round(2)
    g["SFR"] = g["SFR"].round(2)

    write_tsv(out, "tab_d_safr_per_model",
              g[["model", "FFR_rank", "FFR", "SFR_rank", "SFR", "delta_rank"]])


def d_source_type_by_model(am: pd.DataFrame, out: Path) -> None:
    """Paper Table tab_d_source_type_by_model — per-model ST distribution (%) sorted by SFR."""
    sfr = (am.groupby("model_short", sort=False)
             .ssm_score.apply(lambda s: 100.0 * (s <= 2).mean()))
    pivot = (am.groupby(["model_short", "ST_label"]).size()
               .unstack("ST_label", fill_value=0))
    pct = pivot.div(pivot.sum(axis=1), axis=0) * 100.0

    # paper column order
    ordered_cols = [c for c in ST_PAPER_ORDER if c in pct.columns]
    pct = pct[ordered_cols].rename(columns=ST_LABELS).round(2)
    pct["SFR_rank"] = sfr.loc[pct.index].rank(method="min").astype(int)
    pct["model"] = [MODEL_DISPLAY[ms] for ms in pct.index]
    pct = pct.sort_values("SFR_rank").reset_index(drop=True)

    cols = ["model", "SFR_rank"] + [ST_LABELS[c] for c in ordered_cols]
    out_df = pct[cols]

    # totals row
    n_total = len(am)
    totals = {"model": "Total", "SFR_rank": ""}
    for c in ordered_cols:
        totals[ST_LABELS[c]] = round(_share((am.ST_label == c).sum(), n_total), 2)
    out_df = pd.concat([out_df, pd.DataFrame([totals])], ignore_index=True)
    write_tsv(out, "tab_d_source_type_by_model", out_df)


# --------------------------------------------------------------------------- #
# D.4 — Provider, Scale, Reasoning Effects
# --------------------------------------------------------------------------- #

def d_provider_profile(am: pd.DataFrame, out: Path) -> None:
    """Paper Table tab_d_provider_profile — per-provider FFR/SFR/AFR."""
    am = am.assign(provider=am.model_short.map(PROVIDER_OF))
    n_total = len(am)
    rows = []
    for prov in PROVIDER_ORDER:
        sub = am[am.provider == prov]
        if not len(sub): continue
        rows.append({
            "provider": prov,
            "n":        len(sub),
            "share":    round(_share(len(sub), n_total), 1),
            "FFR":      round(100.0 * (sub.asf_score  <= 2).mean(), 1),
            "SFR":      round(100.0 * (sub.ssm_score  <= 2).mean(), 1),
            "AFR":      round(100.0 * (sub.ipam_score <= 2).mean(), 1),
        })
    write_tsv(out, "tab_d_provider_profile", pd.DataFrame(rows))


def d_variance(am: pd.DataFrame, out: Path) -> None:
    """Paper Table tab_d_variance — two-way ANOVA SumSq decomposition."""
    am = am.assign(provider=am.model_short.map(PROVIDER_OF))
    rows = []
    for dim, col in [("IPA score", "ipam_score"),
                     ("AS score",  "asf_score"),
                     ("SS score",  "ssm_score")]:
        prov_means  = am.groupby("provider")[col].mean()
        model_means = am.groupby("model_short")[col].mean()
        ss_between = float(((prov_means - am[col].mean()) ** 2 * am.groupby("provider").size()).sum())
        # Within-provider component = total within-provider variation between models
        within_total = 0.0
        for p, sub in am.groupby("provider"):
            mu_p = sub[col].mean()
            for m, sub2 in sub.groupby("model_short"):
                within_total += len(sub2) * (sub2[col].mean() - mu_p) ** 2
        ss_within = float(within_total)
        total = ss_between + ss_within
        rows.append({
            "metric":          dim,
            "sumsq_between":   round(ss_between, 1),
            "pct_between":     round(100.0 * ss_between / total, 1) if total else 0.0,
            "sumsq_within":    round(ss_within, 1),
            "pct_within":      round(100.0 * ss_within / total, 1) if total else 0.0,
        })
    write_tsv(out, "tab_d_variance", pd.DataFrame(rows))


def _paired_table(am: pd.DataFrame, pairs: list, name: str, out: Path) -> None:
    """Generic paired comparison table (used for both model_size and reasoning)."""
    rows = []
    cit_per_q = (am.groupby(["model_short", "query_id"]).size()
                   .groupby("model_short").mean())
    for prov, m1, m2 in pairs:
        for ms in (m1, m2):
            sub = am[am.model_short == ms]
            rows.append({
                "provider": prov,
                "model":    MODEL_DISPLAY[ms],
                "n":        len(sub),
                "cit_per_q": round(float(cit_per_q.get(ms, 0)), 1),
                "FFR":      round(100.0 * (sub.asf_score <= 2).mean(), 2),
                "SFR":      round(100.0 * (sub.ssm_score <= 2).mean(), 2),
            })
        # Δ row (m1 - m2)
        s1 = am[am.model_short == m1]; s2 = am[am.model_short == m2]
        rows.append({
            "provider": prov,
            "model":    f"Δ ({m1} − {m2})",
            "n":        "",
            "cit_per_q": "",
            "FFR":      round(100.0 * (s1.asf_score <= 2).mean()
                              - 100.0 * (s2.asf_score <= 2).mean(), 2),
            "SFR":      round(100.0 * (s1.ssm_score <= 2).mean()
                              - 100.0 * (s2.ssm_score <= 2).mean(), 2),
        })
    write_tsv(out, name, pd.DataFrame(rows))


def d_model_size_paired(am: pd.DataFrame, out: Path) -> None:
    """Paper Table tab_d_model_size_paired — within-provider model-size pairs."""
    _paired_table(am, MODEL_SIZE_PAIRS, "tab_d_model_size_paired", out)


def d_reasoning_paired(am: pd.DataFrame, out: Path) -> None:
    """Paper Table tab_d_reasoning_paired — reasoning vs non-reasoning pairs."""
    _paired_table(am, REASONING_PAIRS, "tab_d_reasoning_paired", out)


# --------------------------------------------------------------------------- #
# D.5 — Response-Level Failure Exposure
# --------------------------------------------------------------------------- #

def d_response_exposure(am: pd.DataFrame, out: Path) -> None:
    """Paper Table tab_d_response_exposure — citation-level vs response-level rates."""
    rows = []
    for ms, sub in am.groupby("model_short", sort=False):
        if ms not in MODEL_DISPLAY: continue
        # Per-response failure: a response is (query_id, model_short).
        per_resp = sub.groupby("query_id").agg(
            asf_min=("asf_score",  "min"),
            ssm_min=("ssm_score",  "min"),
            ipa_min=("ipam_score", "min"),
            n_cit  =("cit_id",     "count"),
        )
        n_resp = len(per_resp)
        nbar   = float(per_resp.n_cit.mean()) if n_resp else 0.0

        rows.append({
            "model":   MODEL_DISPLAY[ms],
            "nbar":    round(nbar, 1),
            "FFR":     round(100.0 * (sub.asf_score  <= 2).mean(), 1),
            "R_FFR":   round(100.0 * (per_resp.asf_min <= 2).mean(), 1) if n_resp else float("nan"),
            "SFR":     round(100.0 * (sub.ssm_score  <= 2).mean(), 1),
            "R_SFR":   round(100.0 * (per_resp.ssm_min <= 2).mean(), 1) if n_resp else float("nan"),
            "AFR":     round(100.0 * (sub.ipam_score <= 2).mean(), 1),
            "R_AFR":   round(100.0 * (per_resp.ipa_min <= 2).mean(), 1) if n_resp else float("nan"),
            "Any":     round(100.0 * ((per_resp.asf_min <= 2) | (per_resp.ssm_min <= 2)
                                      | (per_resp.ipa_min <= 2)).mean(), 1)
                       if n_resp else float("nan"),
        })

    df = pd.DataFrame(rows).sort_values("Any", ascending=False).reset_index(drop=True)
    write_tsv(out, "tab_d_response_exposure", df)


# --------------------------------------------------------------------------- #
# D.6 — Robustness
# --------------------------------------------------------------------------- #

def d_threshold(am: pd.DataFrame, out: Path) -> None:
    """Paper Table tab_d_threshold — seven ±1 perturbations of the ≤2 failure
    threshold. Kendall τ is computed on per-model and per-category CritVM
    *rates* (not raw counts) against the baseline (≤2,≤2,≤2) configuration.
    """
    if not _HAS_SCIPY:
        LOG.warning("threshold table needs scipy.stats.kendalltau; skipping")
        return

    def _fail_mask(ipa_thr, asf_thr, ss_thr):
        return ((am.ipam_score <= ipa_thr)
                & (am.asf_score  <= asf_thr)
                & (am.ssm_score  <= ss_thr))

    def _rates(mask):
        tmp = am.assign(_f=mask.astype(int))
        return (tmp.groupby("model_short")._f.mean(),
                tmp.groupby("category")._f.mean())

    baseline_mask = _fail_mask(2, 2, 2)
    base_m, base_c = _rates(baseline_mask)

    variants = [
        ("baseline",   2, 2, 2),
        ("as_loose",   2, 3, 2),
        ("as_strict",  2, 1, 2),
        ("ipa_loose",  3, 2, 2),
        ("ss_loose",   2, 2, 3),
        ("ss_strict",  2, 2, 1),
        ("ipa_strict", 1, 2, 2),
    ]
    rows = []
    for name, ipa_t, as_t, ss_t in variants:
        m = _fail_mask(ipa_t, as_t, ss_t)
        n_cv = int(m.sum())
        pct_cv = 100.0 * m.mean()
        rate_m, rate_c = _rates(m)

        common_m = base_m.index.intersection(rate_m.index)
        tau_m = float("nan") if len(common_m) < 2 else \
            _stats.kendalltau(base_m.loc[common_m], rate_m.loc[common_m]).statistic
        common_c = base_c.index.intersection(rate_c.index)
        tau_c = float("nan") if len(common_c) < 2 else \
            _stats.kendalltau(base_c.loc[common_c], rate_c.loc[common_c]).statistic

        rows.append({
            "variant":   name,
            "IPA_le":    ipa_t,
            "ASF_le":    as_t,
            "SS_le":     ss_t,
            "n_critvm":  n_cv,
            "pct":       round(pct_cv, 3),
            "tau_model": round(tau_m, 3),
            "tau_cat":   round(tau_c, 3),
        })
    write_tsv(out, "tab_d_threshold", pd.DataFrame(rows))


def d_temporal(am: pd.DataFrame, out: Path) -> None:
    """Paper Table tab_d_temporal — split at April 3, 2026 (median crawl date).

    Requires ``crawled_at_first`` joined from sources.parquet by main().
    """
    if "crawled_at_first" not in am.columns:
        LOG.warning("crawled_at_first missing; skipping tab_d_temporal")
        return
    ts = pd.to_datetime(am.crawled_at_first, errors="coerce")
    cutoff = pd.Timestamp("2026-04-03")
    early = am[ts < cutoff]
    late  = am[ts >= cutoff]
    rows = []
    for label, sub in [
        (f"First ({ts.min().strftime('%-m/%d')}–{(cutoff - pd.Timedelta(days=1)).strftime('%-m/%d')})", early),
        (f"Second ({cutoff.strftime('%-m/%d')}–{ts.max().strftime('%-m/%d')})", late),
    ]:
        if not len(sub):
            continue
        rows.append({
            "half": label,
            "n":    len(sub),
            "FFR":  round(100.0 * (sub.asf_score  <= 2).mean(), 1),
            "SFR":  round(100.0 * (sub.ssm_score  <= 2).mean(), 1),
            "AFR":  round(100.0 * (sub.ipam_score <= 2).mean(), 1),
        })
    write_tsv(out, "tab_d_temporal", pd.DataFrame(rows))


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #

GENERATORS = [
    ("D.1 pool composition",        d_pool_composition),
    ("D.2 score distribution",      d_score_dist),
    ("D.2 three-axis",              d_three_axis),
    ("D.2 YMYL SFR",                d_ymyl_ssfr),
    ("D.2 YMYL ST composition",     d_ymyl_st),
    ("D.3 SAFR per model",          d_safr_per_model),
    ("D.3 source type by model",    d_source_type_by_model),
    ("D.4 provider profile",        d_provider_profile),
    ("D.4 ANOVA variance",          d_variance),
    ("D.4 model-size paired",       d_model_size_paired),
    ("D.4 reasoning paired",        d_reasoning_paired),
    ("D.5 response exposure",       d_response_exposure),
    ("D.6 threshold sensitivity",   d_threshold),
    ("D.6 temporal stability",      d_temporal),
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True,
                    help="Path or hf:// URL to analysis_master.parquet")
    ap.add_argument("--output-dir", required=True,
                    help="Directory to write tab_d_*.tsv files")
    ap.add_argument("--with-validation", action="store_true",
                    help="Also write Appendix C reliability tables "
                         "(tab_c_judge_kappa, tab_c_validation) from the "
                         "*_human_eval.parquet siblings of --input.")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    LOG.info("loading %s", args.input)
    am = pd.read_parquet(args.input)
    LOG.info("loaded %d rows, %d columns", len(am), len(am.columns))

    # tab_d_temporal needs crawled_at_first; pull it from sibling sources.parquet
    # and left-join on url_id. Look in two places: input's own directory
    # (canonical HF layout) and `parquet/` subdir (pipeline.sh sample-run layout).
    in_path = Path(args.input)
    if in_path.exists():
        sib = in_path.parent / "sources.parquet"
        if not sib.exists():
            sib = in_path.parent / "parquet" / "sources.parquet"
        if sib.exists() and "url_id" in am.columns and "crawled_at_first" not in am.columns:
            try:
                src = pd.read_parquet(sib, columns=["url_id", "crawled_at_first"])
                am = am.merge(src, on="url_id", how="left")
                LOG.info("joined crawled_at_first from %s (na=%d)",
                         sib, int(am.crawled_at_first.isna().sum()))
            except Exception as exc:                  # noqa: BLE001
                LOG.warning("failed to join sources.parquet: %s", exc)

    n_skipped = 0
    for label, fn in GENERATORS:
        try:
            fn(am, out)
        except ValueError as exc:
            # Statistical generators (Kruskal-Wallis, χ², etc.) raise ValueError
            # when the input is too small or has insufficient group variance.
            # Skip and continue so smoke-test runs produce partial output.
            LOG.warning("SKIP %s: %s", label, exc)
            n_skipped += 1
        except Exception as exc:                     # noqa: BLE001
            LOG.error("FAILED %s: %s", label, exc)
            raise

    if args.with_validation:
        from . import human_validate
        in_path = Path(args.input)
        parquet_dir = in_path.parent if in_path.exists() else None
        scoring_dir = Path(__file__).resolve().parents[2] / "scoring_matrices"
        if parquet_dir is None:
            LOG.warning("--with-validation requires a local --input path; skipping")
        else:
            human_validate.run(parquet_dir, scoring_dir, out)

    n_ran = len(GENERATORS) - n_skipped
    LOG.info("done — %d/%d generators ran (%d skipped: insufficient sample)",
             n_ran, len(GENERATORS), n_skipped)


if __name__ == "__main__":
    main()
