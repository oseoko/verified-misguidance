"""Stratified deterministic selection of 3 sample queries from queries.parquet.

Re-run this script (with the same Hugging Face snapshot) to regenerate
``samples/queries.parquet`` and ``samples/queries.jsonl``.

Usage:
    python samples/_select.py /path/to/queries.parquet
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

QI_PICKS = ["QI1", "QI2", "QI4"]   # factoid, explanation, comparison
SEED = 42


def select(queries: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for qi in QI_PICKS:
        sub = queries[queries.QI_label == qi]
        rows.append(sub.sample(1, random_state=SEED).iloc[0])
    return pd.DataFrame(rows).reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("queries_parquet", type=Path)
    args = ap.parse_args()

    out_dir = Path(__file__).resolve().parent
    queries = pd.read_parquet(args.queries_parquet)
    sample  = select(queries)

    sample.to_parquet(out_dir / "queries.parquet", index=False)
    with open(out_dir / "queries.jsonl", "w") as f:
        for row in sample.to_dict(orient="records"):
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"wrote {len(sample)} rows:")
    for _, r in sample.iterrows():
        print(f"  {r.query_id}  {r.QI_label}  {r.site!r}/{r.category!r}")


if __name__ == "__main__":
    main()
