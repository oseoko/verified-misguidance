"""Reproduce paper Appendix C reliability tables from human_eval parquets.

Loads the seven ``*_human_eval.parquet`` files alongside analysis_master
and writes:
    tab_c_judge_kappa.tsv  — Cohen's κ of the LLM judge vs human consensus
                              across 5 dimensions (paper Table c_judge_kappa).
    tab_c_validation.tsv   — ICC(2,k), Pearson r, MAD for the IPA Matrix
                              (30 cells) and the SS Matrix (per-domain +
                              aggregate, paper Table c_validation).
"""
from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from scipy import stats as _stats
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

LOG = logging.getLogger("validate")

DIMS_NOMINAL = ["QI", "SP", "SD", "ST", "ASF"]
DIM_FILES = {d: f"{d.lower()}_human_eval.parquet" for d in DIMS_NOMINAL}


# --------------------------------------------------------------------------- #
# κ helpers
# --------------------------------------------------------------------------- #

def _majority_vote(labels: list[str]) -> str:
    """Majority vote with first-annotator tie-break (matches paper)."""
    c = Counter(labels)
    mx = max(c.values())
    if mx >= 2:
        return [k for k, v in c.items() if v == mx][0]
    return labels[0]


def _cohen_kappa(a: list[str], b: list[str]) -> float:
    """Plain (unweighted) Cohen's κ for nominal labels."""
    cats = sorted(set(a) | set(b))
    n = len(a)
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    ca = Counter(a); cb = Counter(b)
    pe = sum((ca[c] / n) * (cb[c] / n) for c in cats)
    return (po - pe) / (1 - pe) if pe != 1 else float("nan")


def _bootstrap_ci(rng, x: np.ndarray, y: np.ndarray, B: int = 1000,
                  alpha: float = 0.05) -> tuple[float, float]:
    n = len(x)
    out = []
    for _ in range(B):
        idx = rng.integers(0, n, n)
        out.append(_cohen_kappa(list(x[idx]), list(y[idx])))
    lo, hi = np.quantile(out, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def _balanced_accuracy(y_true: list[str], y_pred: list[str]) -> float:
    cats = sorted(set(y_true))
    recalls = []
    for c in cats:
        idx = [i for i, t in enumerate(y_true) if t == c]
        if not idx:
            continue
        rec = sum(1 for i in idx if y_pred[i] == c) / len(idx)
        recalls.append(rec)
    return float(np.mean(recalls)) if recalls else float("nan")


# --------------------------------------------------------------------------- #
# ICC(2,k) helpers
# --------------------------------------------------------------------------- #

def _icc_2k(scores_per_cell: list[list[int]]) -> tuple[float, float, float]:
    """ICC(2,k) two-way random, average measures, absolute agreement,
    plus Wald 95% CI on the F-distribution. Returns (icc, lo, hi)."""
    arr = np.array([list(s) for s in scores_per_cell], dtype=float)
    n, k = arr.shape
    grand = arr.mean()
    msr = (k * ((arr.mean(axis=1) - grand) ** 2).sum()) / (n - 1)
    msc = (n * ((arr.mean(axis=0) - grand) ** 2).sum()) / (k - 1)
    sse = ((arr - arr.mean(axis=1, keepdims=True)
                - arr.mean(axis=0, keepdims=True) + grand) ** 2).sum()
    mse = sse / ((n - 1) * (k - 1))
    icc = (msr - mse) / (msr + (msc - mse) / n)

    if not _HAS_SCIPY:
        return float(icc), float("nan"), float("nan")
    # 95% CI per Shrout & Fleiss (1979) for ICC(2,k)
    F = msr / mse if mse > 0 else float("inf")
    df1 = n - 1
    df2 = (n - 1) * (k - 1)
    F_lo = F / _stats.f.ppf(0.975, df1, df2)
    F_hi = F * _stats.f.ppf(0.975, df2, df1)
    lo = (F_lo - 1) / (F_lo + (msc - mse) / mse) if mse > 0 else float("nan")
    hi = (F_hi - 1) / (F_hi + (msc - mse) / mse) if mse > 0 else float("nan")
    return float(icc), float(lo), float(hi)


def _design_score(matrix_tsv: Path, row_label: str, col_label: str) -> int | None:
    df = pd.read_csv(matrix_tsv, sep="\t")
    if "score" not in df.columns:
        return None
    a, b = df.columns[0], df.columns[1]
    sub = df[(df[a] == row_label) & (df[b] == col_label)]
    if not len(sub):
        return None
    return int(sub.iloc[0]["score"])


# --------------------------------------------------------------------------- #
# Public entry points
# --------------------------------------------------------------------------- #

def write_judge_kappa(parquet_dir: Path, out_dir: Path) -> None:
    """Paper Table tab_c_judge_kappa — κ of LLM judge vs majority-vote human."""
    rng = np.random.default_rng(0)
    rows = []
    for dim in DIMS_NOMINAL:
        path = parquet_dir / DIM_FILES[dim]
        if not path.exists():
            LOG.warning("missing %s; skipping %s", path, dim)
            continue
        df = pd.read_parquet(path)
        h = df.human_labels.apply(lambda a: list(a)).tolist()
        g = df.gpt4o_mini_label.tolist()
        h_maj = [_majority_vote(lbls) for lbls in h]
        n = len(df)
        raw = sum(1 for x, y in zip(h_maj, g) if x == y) / n

        kappa_cons = _cohen_kappa(h_maj, g)

        # κ_pair = mean of pairwise κ between judge and each annotator
        kappa_pair_vals = []
        max_a = max(len(x) for x in h)
        for ai in range(max_a):
            ann = [x[ai] for x in h if ai < len(x)]
            jud = [g[i]   for i, x in enumerate(h) if ai < len(x)]
            if len(set(ann)) > 1:
                kappa_pair_vals.append(_cohen_kappa(ann, jud))
        kappa_pair = float(np.mean(kappa_pair_vals)) if kappa_pair_vals else float("nan")

        bal_acc = _balanced_accuracy(h_maj, g)

        lo, hi = _bootstrap_ci(rng, np.array(h_maj), np.array(g))

        rows.append({
            "dimension":   dim,
            "n":           n,
            "raw_agr":     round(raw, 3),
            "kappa_cons":  round(kappa_cons, 3),
            "kappa_ci":    f"[{lo:.3f}, {hi:.3f}]",
            "kappa_pair":  round(kappa_pair, 3),
            "bal_acc":     round(bal_acc, 3),
        })
    df = pd.DataFrame(rows, columns=["dimension","n","raw_agr","kappa_cons",
                                     "kappa_ci","kappa_pair","bal_acc"])
    df.to_csv(out_dir / "tab_c_judge_kappa.tsv", sep="\t", index=False)
    LOG.info("wrote tab_c_judge_kappa (%d rows)", len(df))


def write_matrix_validation(parquet_dir: Path, scoring_dir: Path,
                            out_dir: Path) -> None:
    """Paper Table tab_c_validation — ICC(2,k) + Pearson r + MAD for IPA, SS."""
    rows = []

    # IPA aggregate
    ipam_path = parquet_dir / "ipam_human_eval.parquet"
    ipam_mat  = scoring_dir / "ipa_matrix.tsv"
    if ipam_path.exists():
        df = pd.read_parquet(ipam_path)
        scores = [list(s) for s in df.annotator_scores]
        icc, lo, hi = _icc_2k(scores)
        consensus = np.array([np.median(s) for s in scores])
        design = np.array([_design_score(ipam_mat, q, sp)
                           for q, sp in zip(df.QI_label, df.SP_label)],
                          dtype=float)
        ok = ~np.isnan(design)
        r = float(np.corrcoef(consensus[ok], design[ok])[0, 1]) if ok.sum() > 1 else float("nan")
        mad = float(np.mean([np.abs(np.array(s) - np.median(s)).mean() for s in scores]))
        rows.append({
            "matrix":    "IPA",
            "row":       "Aggregate (30 cells)",
            "ymyl":      "",
            "n":         10,
            "icc_2k":    round(icc, 3),
            "icc_ci":    f"[{lo:.2f}, {hi:.2f}]",
            "pearson_r": round(r, 3),
            "mad":       round(mad, 3),
        })

    # SS per-domain + aggregate
    ssm_path = parquet_dir / "ssm_human_eval.parquet"
    ssm_mat  = scoring_dir / "ss_matrix.tsv"
    if ssm_path.exists():
        df = pd.read_parquet(ssm_path)
        ymyl = {"SD1", "SD2", "SD3"}
        sd_order = sorted(df.SD_label.unique(),
                          key=lambda s: int(s.replace("SD", "")))
        for sd in sd_order:
            grp = df[df.SD_label == sd]
            scores = [list(s) for s in grp.annotator_scores]
            icc, lo, hi = _icc_2k(scores)
            consensus = np.array([np.median(s) for s in scores])
            design = np.array([_design_score(ssm_mat, sd, st)
                               for st in grp.ST_label], dtype=float)
            ok = ~np.isnan(design)
            r = float(np.corrcoef(consensus[ok], design[ok])[0, 1]) if ok.sum() > 1 else float("nan")
            mad = float(np.mean([np.abs(np.array(s) - np.median(s)).mean() for s in scores]))
            row_label = f"{sd} {grp.SD_name.iloc[0]}" if "SD_name" in grp.columns else sd
            rows.append({
                "matrix":    "SS",
                "row":       row_label,
                "ymyl":      "Y" if sd in ymyl else "",
                "n":         10,
                "icc_2k":    round(icc, 3),
                "icc_ci":    f"[{lo:.2f}, {hi:.2f}]",
                "pearson_r": round(r, 3),
                "mad":       round(mad, 3),
            })
        # Aggregate SS
        scores = [list(s) for s in df.annotator_scores]
        icc, lo, hi = _icc_2k(scores)
        consensus = np.array([np.median(s) for s in scores])
        design = np.array([_design_score(ssm_mat, sd, st)
                           for sd, st in zip(df.SD_label, df.ST_label)],
                          dtype=float)
        ok = ~np.isnan(design)
        r = float(np.corrcoef(consensus[ok], design[ok])[0, 1]) if ok.sum() > 1 else float("nan")
        mad = float(np.mean([np.abs(np.array(s) - np.median(s)).mean() for s in scores]))
        rows.append({
            "matrix":    "SS",
            "row":       "Aggregate (60 cells)",
            "ymyl":      "",
            "n":         100,
            "icc_2k":    round(icc, 3),
            "icc_ci":    f"[{lo:.2f}, {hi:.2f}]",
            "pearson_r": round(r, 3),
            "mad":       round(mad, 3),
        })

    df = pd.DataFrame(rows, columns=["matrix","row","ymyl","n","icc_2k",
                                     "icc_ci","pearson_r","mad"])
    df.to_csv(out_dir / "tab_c_validation.tsv", sep="\t", index=False)
    LOG.info("wrote tab_c_validation (%d rows)", len(df))


def run(parquet_dir: Path, scoring_dir: Path, out_dir: Path) -> None:
    write_judge_kappa(parquet_dir, out_dir)
    write_matrix_validation(parquet_dir, scoring_dir, out_dir)
