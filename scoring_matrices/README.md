# Scoring Matrices

Long-format TSV lookup tables consumed by `src/citetrace/scoring.py`
(re-exported by `src.citetrace.data.build_master`) and the reliability
tables in `src/analysis/{reproduce,human_validate}.py`.

| File             |              Rows | Defined in    | Description                                                                                  |
| ---------------- | ----------------: | ------------- | -------------------------------------------------------------------------------------------- |
| `ipa_matrix.tsv` | 30 (5 QI × 6 SP)  | Appendix §C.2 | IPA[QI][SP] ∈ {1..5}. Judge enum restricts inputs to `QI1..QI5` × `SP1..SP6`. |
| `ss_matrix.tsv`  | 60 (10 SD × 6 ST) | Appendix §C.3 | SS[SD][ST] ∈ {1..5}. Judge enum restricts inputs to `SD1..SD10` × `ST1..ST6`. |

ASF scores are derived directly from the label string
(`ASF1=1, …, ASF5=5`) via `int(ASF_label[3:])` — no separate lookup
file needed.

## Example (Python)

```python
import pandas as pd

ipa = pd.read_csv("ipa_matrix.tsv", sep="\t").set_index(["QI", "SP"])["score"]
ss  = pd.read_csv("ss_matrix.tsv",  sep="\t").set_index(["SD", "ST"])["score"]

am = pd.read_parquet("../data/analysis_master.parquet")
am["ipam_score"] = am.set_index(["QI_label", "SP_label"]).index.map(ipa)
am["ssm_score"]  = am.set_index(["SD_label", "ST_label"]).index.map(ss)
am["asf_score"]  = am["ASF_label"].str[3:].astype(int)
```
