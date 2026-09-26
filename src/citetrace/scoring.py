"""IPA and SS matrix score lookup (paper §C.2, §C.3).

These are pure functions that map taxonomy labels to integer scores using
the CiteTrace scoring matrices. Useful if you want to recompute scores from
raw labels without using the precomputed columns in analysis_master.

Example:
    >>> from citetrace.scoring import ipam_score, ssm_score, asf_score
    >>> ipam_score("QI3", "SP1")
    1
    >>> ssm_score("SD7", "ST5")
    3
    >>> asf_score("ASF2")
    2
"""
from __future__ import annotations

from functools import cache
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCORING_DIR = _REPO_ROOT / "scoring_matrices"


@cache
def _load_matrix(kind: str) -> dict[tuple[str, str], int]:
    """Load and cache an IPA or SS matrix from scoring_matrices/{kind}_matrix.tsv.

    Expects 3-col long format: (row_key, col_key, score) with header
    ending in "score".
    """
    path = _SCORING_DIR / f"{kind}_matrix.tsv"
    if not path.exists():
        raise FileNotFoundError(
            f"{kind} matrix not found at {path}. "
            "Ensure scoring_matrices/ exists and contains "
            "ipa_matrix.tsv and ss_matrix.tsv."
        )

    out: dict[tuple[str, str], int] = {}
    with path.open() as f:
        header = f.readline().rstrip("\n").split("\t")
        if len(header) != 3 or header[2].lower() != "score":
            raise ValueError(
                f"unexpected header in {path}: {header}; "
                f"expected 3-col long format (row_key, col_key, score)"
            )
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) == 3 and parts[2].strip():
                out[(parts[0], parts[1])] = int(parts[2])
    return out


def ipam_score(qi_label: str, sp_label: str) -> int:
    """IPA matrix score for (Query Intent, Source Purpose) pair. Returns 1-5.

    Args:
        qi_label: Query Intent label, one of QI1..QI5.
        sp_label: Source Purpose label, one of SP1..SP6.

    Returns:
        Integer score in [1, 5] from the IPA 5x6 matrix (§C.2.1).

    Raises:
        KeyError: if (qi_label, sp_label) is not a valid combination.
    """
    return _load_matrix("ipa")[(qi_label, sp_label)]


def ssm_score(sd_label: str, st_label: str) -> int:
    """SS matrix score for (Source Domain, Source Type) pair. Returns 1-5.

    Args:
        sd_label: Source Domain label, one of SD1..SD10.
        st_label: Source Type label, one of ST1..ST6.

    Returns:
        Integer score in [1, 5] from the SS 10x6 matrix (§C.3.1).

    Raises:
        KeyError: if (sd_label, st_label) is not a valid combination.
    """
    return _load_matrix("ss")[(sd_label, st_label)]


def asf_score(asf_label: str) -> int:
    """Parse ASF_label string ('ASF3') into integer score (3).

    Args:
        asf_label: One of ASF1..ASF5.

    Returns:
        Integer score in [1, 5].
    """
    return int(asf_label[3:])
