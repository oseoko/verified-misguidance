"""Stratified sampling for the ASF (Answer-Source Fidelity) review pool.

Reads ``data/analysis_master.parquet`` and draws 200 rows:

  Layer 1 — per-model verdict balance: 10 models × (ASF5 × 10 + ASF1 × 4) = 140
  Layer 2 — distortion subtype balance: ASF3 × 20 + ASF4 × 20 + ASF2 × 20 = 60

Output JSONL is the pre-annotation form of
``schemas/label_human_eval.schema.json``; ``human_labels`` is filled
by the downstream annotation step.

    python -m scripts.sample_asf_human_eval \\
        --master-parquet data/analysis_master.parquet \\
        --output-dir     data/asf_samples \\
        --seed           42
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd


MODELS = [
    "claude-haiku",
    "claude-sonnet",
    "gemini-flash",
    "gemini-pro",
    "gpt-5",
    "gpt-5-mini",
    "grok-nr",
    "grok-r",
    "sonar",
    "sonar-rp",
]

# Per-model: ASF5=Supported, ASF1=Fabricated
LAYER1 = {"ASF5": 10, "ASF1": 4}

# Distortion subtypes: ASF3=Contradicted, ASF4=Amplified, ASF2=Misattributed
LAYER2 = {"ASF3": 20, "ASF4": 20, "ASF2": 20}


def sample_layer1(rows: list[dict], seed: int) -> list[dict]:
    rng = random.Random(seed)
    by_key = defaultdict(list)
    for r in rows:
        if r.get("ASF_label") in LAYER1:
            by_key[(r.get("model_short"), r["ASF_label"])].append(r)
    out: list[dict] = []
    for model in MODELS:
        for label, n in LAYER1.items():
            pool = by_key.get((model, label), [])
            if len(pool) <= n:
                out.extend(pool)
            else:
                out.extend(rng.sample(pool, n))
    return out


def sample_layer2(rows: list[dict], seed: int) -> list[dict]:
    rng = random.Random(seed + 1)
    by_key = defaultdict(list)
    for r in rows:
        if r.get("ASF_label") in LAYER2:
            by_key[r["ASF_label"]].append(r)
    out: list[dict] = []
    for label, n in LAYER2.items():
        pool = by_key[label]
        out.extend(pool if len(pool) <= n else rng.sample(pool, n))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--master-parquet", required=True, type=Path,
                    help="Path to analysis_master.parquet")
    ap.add_argument("--output-dir",     required=True, type=Path)
    ap.add_argument("--seed",           type=int, default=42)
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    am = pd.read_parquet(args.master_parquet)
    rows = am.to_dict(orient="records")
    print(f"analysis_master: {len(rows):,} rows")

    layer1 = sample_layer1(rows, args.seed)
    layer2 = sample_layer2(rows, args.seed)
    print(f"  layer1 (model × verdict): {len(layer1)}")
    print(f"  layer2 (distortion):      {len(layer2)}")
    print(f"  total: {len(layer1) + len(layer2)}")

    samples = layer1 + layer2
    out = args.output_dir / "asf.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for i, r in enumerate(samples, start=1):
            f.write(json.dumps({
                "item_id":          f"ASF{i}",
                "task":             "ASF",
                "cit_id":           int(r["cit_id"]) if pd.notna(r.get("cit_id")) else None,
                "gpt4o_mini_label": r.get("ASF_label"),
            }, ensure_ascii=False) + "\n")

    summary = {
        "total":  len(samples),
        "layer1": dict(Counter((r.get("model_short"), r["ASF_label"]) for r in layer1)),
        "layer2": dict(Counter(r["ASF_label"] for r in layer2)),
    }
    (args.output_dir / "asf_validation.json").write_text(
        json.dumps({k: {f"{a} | {b}": v for (a, b), v in d.items()} if k == "layer1" else d
                    for k, d in summary.items()}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
