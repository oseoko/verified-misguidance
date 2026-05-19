# CiteTrace

> Verified Misguidance: Measuring Structural Citation
> Failures in Search-Augmented LLMs

- **Dataset (HF)**: https://huggingface.co/datasets/oseoko/citetrace-vm
- **License**: Code MIT, Data CC BY 4.0 / CC BY-SA 4.0 (see paper Appendix E.1.3)

---

## Quickstart — Reproduce paper §D

No API keys, no Hugging Face download. The repo ships eleven of the
twelve HF parquets directly under `data/` (only `model_responses.parquet`
is omitted at 212 MB; everything needed for analysis is bundled).

```bash
git clone <repo> && cd CiteTrace
pip install -r requirements.txt

# §D: 14 tables + qualitative cases
python -m src.analysis.reproduce \
    --input      data/analysis_master.parquet \
    --output-dir results/d_tables

# §D + Appendix C reliability (κ, ICC) from human-eval parquets
python -m src.analysis.reproduce \
    --input          data/analysis_master.parquet \
    --output-dir     results/d_tables \
    --with-validation
```

Output: 14 §D TSVs (`tab_d_*.tsv`). With `--with-validation`,
additionally `tab_c_judge_kappa.tsv` and `tab_c_validation.tsv`
(Appendix C: judge κ + matrix ICC, computed from the bundled
`*_human_eval.parquet` files).

> Pinned: Python 3.12 · `gpt-4o-mini-2024-07-18` judge (temp=0) ·
> 10 search models (`src/lib/config.MODEL_REGISTRY`) · every I/O is
> JSON-Schema-validated.

---

## Repository layout

```
CiteTrace/
├── data/                       11 of 12 HF parquets bundled (everything for analysis)
│   ├── analysis_master.parquet  fully-hydrated table for §D reproduction
│   ├── citations.parquet
│   ├── sources.parquet
│   ├── queries.parquet
│   ├── {asf,qi,sd,sp,st}_human_eval.parquet
│   ├── ipam_human_eval.parquet, ssm_human_eval.parquet
│   └── (model_responses.parquet — fetch from HF if needed)
├── src/
│   ├── citetrace/   scoring.py, data.py (build_master), export.py
│   ├── search/      LLM search + crawl (§B.4-B.6) — run.py, extractors.py
│   ├── judge/       Taxonomy classification (§C.1) — prompts.py, runner.py, run.py
│   ├── analysis/    reproduce.py (§D), human_validate.py (Appendix C)
│   └── lib/         crawler, llm/, config, io, log, models, paths, sites
├── scoring_matrices/  IPA 5×6, SS 10×6 lookup TSVs
├── schemas/           JSON Schemas for every released table
├── samples/           3 stratified sample queries (Q07974, Q00774, Q01293)
└── scripts/
    ├── pipeline.sh             End-to-end driver (search | judge | export | master | d | all)
    ├── sample_human_eval.py    Stratified sampling — QI/SP/SD/ST × 200 each
    └── sample_asf_human_eval.py  Stratified sampling — ASF × 200 (model × verdict + distortion subtype)
```

---

## End-to-end pipeline (`scripts/pipeline.sh`)

The dataset-construction flow `queries → §D tables` is implemented as
five stages plus an `all` arm that runs them in sequence. A 3-row
sample query set ships in `samples/queries.{parquet,jsonl}` so the
whole pipeline can be exercised without an HF download.

| Stage | Command | Reads | Writes |
|---|---|---|---|
| `search` | `python -m src.search.run` | `queries.{jsonl,parquet}` | `search/{citations,sources,contents,model_responses}.jsonl` |
| `judge` | `python -m src.judge.run` | `search/*.jsonl` | `judge/{queries,sources,citations}.judged.jsonl` |
| `export` | `python -m src.citetrace.export` | `judge/*.judged.jsonl` + `search/model_responses.jsonl` | `parquet/{queries,sources,citations,model_responses}.parquet` |
| `master` | `src.citetrace.data.build_master` (inline) | `parquet/*.parquet` | `analysis_master.parquet` |
| `d` | `python -m src.analysis.reproduce` | `analysis_master.parquet` | `d_tables/tab_d_*.tsv` |
| `all` | runs the five above in order | — | full output tree |

For reproducing the §D tables themselves, use the bundled
`data/analysis_master.parquet` via the Quickstart command above — that
path is byte-deterministic. The end-to-end driver below exists for
running the search→judge→export→master pipeline on new query sets.

```bash
cp .env.example .env                                          # fill in API keys

# 3-query smoke test on a single model (~$0.05).
# §D tables that compare across models (e.g. `tab_d_three_axis`) are
# undefined on a single group and are skipped automatically.
bash scripts/pipeline.sh all sample                           # → data/sample_run/sample/

# Custom queries × all 10 models — required for full §D table coverage.
QUERIES=path/to/queries.parquet \
  MODELS="gpt-5-2025-08-07 gpt-5-mini-2025-08-07 \
          claude-sonnet-4-6 claude-haiku-4-5-20251001 \
          gemini-3-flash-preview gemini-3.1-pro-preview \
          grok-4-1-fast-non-reasoning grok-4-1-fast-reasoning \
          sonar sonar-reasoning-pro" \
  bash scripts/pipeline.sh all my_run

# Individual stages re-run the same way:
bash scripts/pipeline.sh {search|judge|export|master|d} my_run
```

Output layout under `data/sample_run/<run_id>/`:

| Path | Schema |
|---|---|
| `parquet/{queries,sources,citations,model_responses}.parquet` | matches HF dataset |
| `analysis_master.parquet` (20 cols) | matches `data/analysis_master.parquet` |
| `d_tables/tab_d_*.tsv` | matches paper §D (requires multi-model run) |

### Configuration

| Var | Default | Effect |
|---|---|---|
| `QUERIES` | `samples/queries.jsonl` | input query set (`.parquet` or `.jsonl`) |
| `MODELS` | `claude-haiku-4-5-20251001` | space-separated model ids |
| `JUDGE_CONCURRENCY` | `20` | judge-stage semaphore |

API keys (set in `.env`; only the providers you call need a value):

| Variable | Used for |
| --- | --- |
| `OPENAI_API_KEY` | search (`gpt-*`), judge |
| `ANTHROPIC_API_KEY` | search (`claude-*`) |
| `GOOGLE_API_KEY` | search (`gemini-*`) |
| `XAI_API_KEY` | search (`grok-*`) |
| `PERPLEXITY_API_KEY` | search (`sonar*`) |

The crawler respects each host's `robots.txt` (RFC 9309). Gemini
grounding URLs (`vertexaisearch.cloud.google.com/grounding-api-redirect/...`)
are resolved to their target via a single redirect-follow before the
target host's `robots.txt` is consulted.

---

## License

| Artifact | License | Origin / Rationale |
|---|---|---|
| Query set (11,200 Stack Exchange titles) | CC BY-SA 4.0 | Stack Exchange Data Dump |
| Model response texts & `cited_sentence` extracts | CC BY 4.0 | Per-provider API ToS |
| Taxonomy labels (QI / SP / SD / ST / ASF) | CC BY 4.0 | Research-original output |
| IPA and SS matrices | CC BY 4.0 | Research-original output |
| Aggregate analysis tables (`analysis_master`, etc.) | CC BY 4.0 | Research-original output |
| Collection and analysis code | MIT | Permissive open-source |
| Crawled source contents | Not redistributed | Per-domain ToS review infeasible |
