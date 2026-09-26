"""Centralised path resolution for CiteTrace.

Layout (after ``git clone``):

    CiteTrace/              <- ROOT (repo root = Python package root)
    ├── .env                <- ENV_FILE (copied from .env.example)
    ├── data/               <- DATA
    ├── schemas/
    ├── scoring_matrices/
    └── src/
"""
from __future__ import annotations

from pathlib import Path

ROOT: Path = Path(__file__).resolve().parents[2]

DATA: Path = ROOT / "data"
SCHEMAS: Path = ROOT / "schemas"
SCORING_MATRICES: Path = ROOT / "scoring_matrices"

LOG_DIR: Path = ROOT / "logs"

ENV_FILE: Path = ROOT / ".env"


def schema_path(name: str) -> Path:
    """Resolve a schema by stem, e.g. schema_path('queries')."""
    return SCHEMAS / f"{name}.schema.json"


__all__ = [
    "ROOT",
    "DATA",
    "SCHEMAS",
    "SCORING_MATRICES",
    "LOG_DIR",
    "ENV_FILE",
    "schema_path",
]
