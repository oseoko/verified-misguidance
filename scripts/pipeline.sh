#!/usr/bin/env bash
# CiteTrace end-to-end pipeline driver.
#
# Stages:
#   search   LLM search + crawl                   -> *.jsonl (search/)
#   judge    OpenAI judge over 5 axes             -> *.judged.jsonl (judge/)
#   export   JSONL stages -> HF-format parquet    -> parquet/
#   master   parquet 4-tuple -> analysis_master   -> master/
#   d        analysis_master -> §D table TSVs    -> d_tables/
#
#   all      run search -> judge -> export -> master -> d in one go
#
# Usage:
#   bash scripts/pipeline.sh search  <run_id> [extra search args ...]
#   bash scripts/pipeline.sh judge   <run_id>
#   bash scripts/pipeline.sh export  <run_id>
#   bash scripts/pipeline.sh master  <run_id>
#   bash scripts/pipeline.sh d       <run_id>
#   bash scripts/pipeline.sh all     <run_id>
#
# Default models: claude-haiku-4-5-20251001 (override via MODELS env).
# All stages hit live provider APIs; populate .env first.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY="${PY:-python}"
RUN_ID="${2:-sample}"
RUN_DIR="data/sample_run/${RUN_ID}"
QUERIES="${QUERIES:-samples/queries.jsonl}"
MODELS="${MODELS:-claude-haiku-4-5-20251001}"

mode="${1:-}"

case "$mode" in
  search)
    out="${RUN_DIR}/search"
    $PY -m src.search.run \
        --input  "$QUERIES" \
        --models $MODELS \
        --output "$out" \
        --save-content
    echo "OK - search   -> $out"
    ;;

  judge)
    src_dir="${RUN_DIR}/search"
    out="${RUN_DIR}/judge"
    $PY -m src.judge.run \
        --concurrency   "${JUDGE_CONCURRENCY:-20}" \
        --queries       "$QUERIES" \
        --sources       "${src_dir}/sources.jsonl" \
        --citations     "${src_dir}/citations.jsonl" \
        --content-jsonl "${src_dir}/contents.jsonl" \
        --output-dir    "$out"
    echo "OK - judge    -> $out"
    ;;

  export)
    out="${RUN_DIR}/parquet"
    $PY -m src.citetrace.export \
        --jsonl-dir "$RUN_DIR" \
        --out-dir   "$out"
    echo "OK - export   -> $out"
    ;;

  master)
    in_dir="${RUN_DIR}/parquet"
    out="${RUN_DIR}/analysis_master.parquet"
    $PY -c "
import pandas as pd, sys
sys.path.insert(0, '.')
from src.citetrace.data import build_master
q = pd.read_parquet('${in_dir}/queries.parquet')
s = pd.read_parquet('${in_dir}/sources.parquet')
c = pd.read_parquet('${in_dir}/citations.parquet')
am = build_master(q, s, c)
am.to_parquet('${out}', index=False)
print(f'  wrote {len(am)} rows -> ${out}')
"
    echo "OK - master   -> $out"
    ;;

  d)
    in_path="${RUN_DIR}/analysis_master.parquet"
    out="${RUN_DIR}/d_tables"
    $PY -m src.analysis.reproduce \
        --input      "$in_path" \
        --output-dir "$out"
    echo "OK - d_tables -> $out"
    ;;

  all)
    "$BASH_SOURCE" search "$RUN_ID"
    "$BASH_SOURCE" judge  "$RUN_ID"
    "$BASH_SOURCE" export "$RUN_ID"
    "$BASH_SOURCE" master "$RUN_ID"
    "$BASH_SOURCE" d      "$RUN_ID"
    echo
    echo "ALL stages complete -> $RUN_DIR"
    ;;

  *)
    echo "unknown mode: $mode" >&2
    echo "usage: $0 {search|judge|export|master|d|all} <run_id>" >&2
    exit 1
    ;;
esac
