"""Export human-eval tables without judge reasoning; refresh derived summaries.

Writes a separate directory and never edits the input snapshot. Expert
scores and classification labels are unchanged; the standard median for
an even-sized panel is the mean of the two central scores.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def export(data_dir: Path, output_dir: Path) -> list[dict]:
    if data_dir.resolve() == output_dir.resolve():
        raise ValueError("output-dir must differ from data-dir")
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for task in ("qi", "sp", "sd", "st", "asf", "ipam", "ssm"):
        name = f"{task}_human_eval.parquet"
        table = pq.read_table(data_dir / name)
        changes = {}
        if task in ("ipam", "ssm"):
            scores = table["annotator_scores"].to_pylist()
            for column, values in (
                ("mean", [float(np.mean(s)) for s in scores]),
                ("median", [float(np.median(s)) for s in scores]),
                ("n", [len(s) for s in scores]),
            ):
                changes[column] = sum(a != b for a, b in zip(table[column].to_pylist(), values))
                table = table.set_column(table.schema.get_field_index(column), column,
                                         pa.array(values))
        elif "gpt4o_mini_reasoning" in table.column_names:
            table = table.drop(["gpt4o_mini_reasoning"])
            changes["removed_field"] = "gpt4o_mini_reasoning"
        # Drop stale pandas metadata that may still describe removed columns.
        table = table.replace_schema_metadata(None)
        target = output_dir / name
        pq.write_table(table, target)
        records.append({"file": name, "rows": table.num_rows,
                        "changes": changes,
                        "sha256": hashlib.sha256(target.read_bytes()).hexdigest()})
    (output_dir / "manifest.json").write_text(json.dumps(records, indent=2) + "\n")
    schema_path = Path(__file__).resolve().parents[1] / "schemas/label_human_eval.schema.json"
    schema = json.loads(schema_path.read_text())
    schema["required"] = [key for key in schema["required"] if key != "gpt4o_mini_reasoning"]
    schema["properties"].pop("gpt4o_mini_reasoning", None)
    schema["description"] += " Public export omits the LLM judge reasoning field."
    (output_dir / "label_human_eval.schema.json").write_text(json.dumps(schema, indent=2) + "\n")
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    export(args.data_dir, args.output_dir)


if __name__ == "__main__":
    main()
