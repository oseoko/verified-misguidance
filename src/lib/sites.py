"""Stack Exchange site topology used in CiteTrace.

Single source of truth for the 28 sites and their two parallel groupings:

* SE official 6-category (already present in the released parquets as the
  ``category`` column).
* Paper 4-domain (used in §D analyses: Technology / Science / Public /
  Work). Not in the parquets — this module is the canonical mapping for
  users who want to reproduce or extend the 4-domain analyses.
"""
from __future__ import annotations

from typing import Final


# slug -> SE official site name
SLUG_TO_NAME: Final[dict[str, str]] = {
    "academia": "Academia",
    "ai": "Artificial Intelligence",
    "aviation": "Aviation",
    "bicycles": "Bicycles",
    "bitcoin": "Bitcoin",
    "chemistry": "Chemistry",
    "cooking": "Seasoned Advice",
    "dba": "Database Administrators",
    "diy": "Home Improvement",
    "earthscience": "Earth Science",
    "economics": "Economics",
    "health": "Medical Sciences",
    "law": "Law",
    "mechanics": "Motor Vehicle Maintenance & Repair",
    "money": "Personal Finance & Money",
    "outdoors": "The Great Outdoors",
    "parenting": "Parenting",
    "pets": "Pets",
    "physics": "Physics",
    "politics": "Politics",
    "quant": "Quantitative Finance",
    "security": "Information Security",
    "skeptics": "Skeptics",
    "softwareengineering": "Software Engineering",
    "stackoverflow": "Stack Overflow",
    "stats": "Cross Validated",
    "travel": "Travel",
    "workplace": "The Workplace",
}

NAME_TO_SLUG: Final[dict[str, str]] = {v: k for k, v in SLUG_TO_NAME.items()}

# slug -> 4-domain (paper §D analysis grouping)
SLUG_TO_DOMAIN: Final[dict[str, str]] = {
    "stackoverflow": "Technology",
    "security": "Technology",
    "softwareengineering": "Technology",
    "dba": "Technology",
    "ai": "Technology",
    "stats": "Technology",
    "physics": "Science",
    "chemistry": "Science",
    "economics": "Science",
    "quant": "Science",
    "earthscience": "Science",
    "aviation": "Public",
    "diy": "Public",
    "cooking": "Public",
    "travel": "Public",
    "outdoors": "Public",
    "bicycles": "Public",
    "parenting": "Public",
    "health": "Public",
    "mechanics": "Public",
    "pets": "Public",
    "money": "Work",
    "law": "Work",
    "bitcoin": "Work",
    "politics": "Work",
    "workplace": "Work",
    "academia": "Work",
    "skeptics": "Work",
}

NAME_TO_DOMAIN: Final[dict[str, str]] = {
    SLUG_TO_NAME[s]: d for s, d in SLUG_TO_DOMAIN.items()
}

DOMAIN_ORDER: Final[list[str]] = ["Technology", "Science", "Public", "Work"]

DOMAIN_SITES_SLUG: Final[dict[str, list[str]]] = {
    d: [s for s, dd in SLUG_TO_DOMAIN.items() if dd == d] for d in DOMAIN_ORDER
}


def domain_of(site: str) -> str:
    """Return paper §D 4-domain for either a slug or an official site name."""
    if site in NAME_TO_DOMAIN:
        return NAME_TO_DOMAIN[site]
    if site in SLUG_TO_DOMAIN:
        return SLUG_TO_DOMAIN[site]
    raise KeyError(f"unknown site: {site!r}")


__all__ = [
    "SLUG_TO_NAME",
    "NAME_TO_SLUG",
    "SLUG_TO_DOMAIN",
    "NAME_TO_DOMAIN",
    "DOMAIN_ORDER",
    "DOMAIN_SITES_SLUG",
    "domain_of",
]
