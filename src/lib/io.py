"""Schema-validated I/O helpers.

The pipeline is enforced by ``schemas/*.schema.json``. All readers and
writers route through this module so that:

1. Column / field names cannot drift between stages.
2. Failures are loud and immediate (``ValidationError`` includes the row
   index and the offending field).
3. Stage outputs are uniformly JSONL (newline-delimited JSON), which is
   diff-friendly and streamable.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List

from jsonschema import Draft202012Validator

from . import paths

_SCHEMA_CACHE: Dict[str, Draft202012Validator] = {}


def load_schema(name: str) -> Draft202012Validator:
    """Lazy-load and cache a JSON-schema validator by stem.

    Example: ``load_schema('queries')``.
    """
    if name in _SCHEMA_CACHE:
        return _SCHEMA_CACHE[name]
    p = paths.schema_path(name)
    if not p.exists():
        raise FileNotFoundError(f"Schema not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        schema = json.load(f)
    validator = Draft202012Validator(schema)
    _SCHEMA_CACHE[name] = validator
    return validator


def validate_row(name: str, row: Dict[str, Any], *, idx: int = -1) -> None:
    """Raise a clean ValueError (with row index) if ``row`` doesn't match."""
    validator = load_schema(name)
    errors = sorted(validator.iter_errors(row), key=lambda e: e.path)
    if not errors:
        return
    msgs = [
        f"  - {'/'.join(str(p) for p in e.absolute_path) or '<root>'}: {e.message}"
        for e in errors
    ]
    where = f"row #{idx} " if idx >= 0 else ""
    raise ValueError(
        f"Schema '{name}' validation failed for {where}({len(errors)} error(s)):\n"
        + "\n".join(msgs)
    )


def write_jsonl(
    path: Path,
    rows: Iterable[Dict[str, Any]],
    *,
    schema: str | None = None,
) -> int:
    """Write rows as JSONL. Returns row count.

    If ``schema`` is given, every row is validated before write.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for i, row in enumerate(rows):
            if schema is not None:
                validate_row(schema, row, idx=i)
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            f.write("\n")
            n += 1
    return n


def read_jsonl(
    path: Path,
    *,
    schema: str | None = None,
) -> Iterator[Dict[str, Any]]:
    """Stream JSONL. Validates each row against ``schema`` if given."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"JSONL not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if schema is not None:
                validate_row(schema, row, idx=i)
            yield row


def read_jsonl_list(path: Path, *, schema: str | None = None) -> List[Dict[str, Any]]:
    """Materialise read_jsonl into a list (small files only)."""
    return list(read_jsonl(path, schema=schema))


__all__ = [
    "load_schema",
    "validate_row",
    "write_jsonl",
    "read_jsonl",
    "read_jsonl_list",
]
