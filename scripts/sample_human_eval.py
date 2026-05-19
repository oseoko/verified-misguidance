"""Stratified sampling for the QI / SP / SD / ST human-eval pools.

Reads ``data/analysis_master.parquet``, deduplicates per axis subject
(query_id for QI; url_id for SP/SD/ST), and draws 200 rows stratified
by the axis label with greedy diversity over secondary axes (category
for QI; the other two label axes for SP/SD/ST).

Output JSONL is the pre-annotation form of
``schemas/label_human_eval.schema.json``; ``human_labels`` is filled
by the downstream annotation step.

    python -m scripts.sample_human_eval \\
        --master-parquet data/analysis_master.parquet \\
        --output-dir     data/he_samples \\
        --seed           42
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
from collections import Counter
from pathlib import Path

import pandas as pd


ALLOC = {
    "QI": {f"QI{i}": 40 for i in range(1, 6)},                                   # 5 × 40 = 200
    "ST": {"ST1": 33, "ST2": 33, "ST3": 34, "ST4": 34, "ST5": 33, "ST6": 33},    # 200
    "SD": {f"SD{i}": 20 for i in range(1, 11)},                                  # 10 × 20 = 200
    "SP": {"SP1": 33, "SP2": 34, "SP3": 33, "SP4": 33, "SP5": 34, "SP6": 33},    # 200
}

PRIMARY_LABEL = {"QI": "QI_label", "ST": "ST_label",
                 "SD": "SD_label", "SP": "SP_label"}

SECONDARY = {"QI": ["category"],
             "ST": ["SD_label", "SP_label"],
             "SD": ["ST_label", "SP_label"],
             "SP": ["ST_label", "SD_label"]}

DEDUP_KEY = {"QI": "query_id", "ST": "url_id", "SD": "url_id", "SP": "url_id"}


def axis_pool(am: pd.DataFrame, axis: str) -> list[dict]:
    """Dedup by subject id and keep first occurrence."""
    key = DEDUP_KEY[axis]
    cols = list({key, PRIMARY_LABEL[axis], *SECONDARY[axis]})
    sub = am[cols + [c for c in [key] if c not in cols]].drop_duplicates(subset=[key])
    return sub.to_dict(orient="records")


def greedy_sample(pool: list[dict], primary: str, alloc: dict[str, int],
                  secondary: list[str], seed: int) -> list[dict]:
    selected: list[dict] = []
    for cell_idx, (label, n) in enumerate(alloc.items()):
        rng = random.Random(seed * 1000 + cell_idx)
        bucket = [r for r in pool if r.get(primary) == label]
        counters = {c: Counter() for c in secondary}
        rng.shuffle(bucket)
        for _ in range(n):
            if not bucket:
                break
            best_i, best_s = 0, -1.0
            for i, row in enumerate(bucket):
                s = sum(1.0 / (counters[c][row.get(c)] + 1) for c in secondary)
                s += rng.random() * 0.1
                if s > best_s:
                    best_s, best_i = s, i
            chosen = bucket.pop(best_i)
            selected.append(chosen)
            for c in secondary:
                counters[c][chosen.get(c)] += 1
    return selected


def report(sample: list[dict], primary: str, alloc: dict[str, int],
           secondary: list[str]) -> dict:
    n = len(sample)
    pc = Counter(r.get(primary) for r in sample)
    rep = {"n": n,
           "primary_counts": dict(sorted(pc.items())),
           "primary_match_alloc": dict(pc) == alloc,
           "secondary": {}}
    for c in secondary:
        cnt = Counter(r.get(c) for r in sample)
        vals = list(cnt.values())
        rep["secondary"][c] = {
            "distinct":  sum(1 for v in vals if v > 0),
            "min":       min(vals) if vals else 0,
            "max":       max(vals) if vals else 0,
            "std":       round(statistics.pstdev(vals), 3) if len(vals) > 1 else 0.0,
            "max_dev":   round(max(abs(v / n - 1 / len(cnt)) for v in vals) if cnt else 0.0, 4),
        }
    return rep


def write_axis(axis: str, sample: list[dict], rep: dict, out_dir: Path) -> None:
    key       = DEDUP_KEY[axis]
    label_col = PRIMARY_LABEL[axis]
    out = out_dir / f"{axis.lower()}.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for i, r in enumerate(sample, start=1):
            row = {"item_id": f"{axis}{i}", "task": axis,
                   key: r.get(key),
                   "gpt4o_mini_label": r.get(label_col)}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    (out_dir / f"{axis.lower()}_validation.json").write_text(
        json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--master-parquet", required=True, type=Path,
                    help="Path to analysis_master.parquet")
    ap.add_argument("--output-dir",     required=True, type=Path)
    ap.add_argument("--seed",           type=int, default=42)
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    am = pd.read_parquet(args.master_parquet)
    print(f"analysis_master: {len(am):,} rows")

    for axis in ("QI", "ST", "SD", "SP"):
        pool = axis_pool(am, axis)
        pool = [r for r in pool if r.get(PRIMARY_LABEL[axis]) in ALLOC[axis]]
        sample = greedy_sample(pool, PRIMARY_LABEL[axis], ALLOC[axis],
                               SECONDARY[axis], args.seed)
        rep = report(sample, PRIMARY_LABEL[axis], ALLOC[axis], SECONDARY[axis])
        write_axis(axis, sample, rep, args.output_dir)
        ok = "✓" if rep["primary_match_alloc"] else "✗"
        print(f"  {axis}: pool={len(pool):>5}  n={rep['n']:>3}  primary={ok}")


if __name__ == "__main__":
    main()
