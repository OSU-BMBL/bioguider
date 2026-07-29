"""Invariant tests for ERROR_CATEGORIES partitioning.

CONTENT, HYGIENE, UNSCORABLE must be pairwise disjoint and their union must
equal ALL_ERROR_CATEGORIES. If someone adds a new category without updating
the groups, these tests fail loudly at CI time.
"""
from __future__ import annotations

from bioguider.managers.config import (
    ALL_ERROR_CATEGORIES,
    CONTENT_CATEGORIES,
    HYGIENE_CATEGORIES,
    UNSCORABLE_CATEGORIES,
    SCORABLE_CATEGORIES,
)


def test_groups_disjoint():
    assert not (CONTENT_CATEGORIES & HYGIENE_CATEGORIES), (
        f"CONTENT and HYGIENE overlap: {CONTENT_CATEGORIES & HYGIENE_CATEGORIES}"
    )
    assert not (CONTENT_CATEGORIES & UNSCORABLE_CATEGORIES), (
        f"CONTENT and UNSCORABLE overlap: {CONTENT_CATEGORIES & UNSCORABLE_CATEGORIES}"
    )
    assert not (HYGIENE_CATEGORIES & UNSCORABLE_CATEGORIES), (
        f"HYGIENE and UNSCORABLE overlap: {HYGIENE_CATEGORIES & UNSCORABLE_CATEGORIES}"
    )


def test_groups_cover_all_categories():
    union = CONTENT_CATEGORIES | HYGIENE_CATEGORIES | UNSCORABLE_CATEGORIES
    all_cats = set(ALL_ERROR_CATEGORIES)
    missing = all_cats - union
    extra = union - all_cats
    assert not missing, f"categories not assigned to any group: {missing}"
    assert not extra, f"unknown categories in groups: {extra}"


def test_content_plus_hygiene_equals_scorable():
    """CONTENT + HYGIENE should equal the historical SCORABLE_CATEGORIES."""
    combined = CONTENT_CATEGORIES | HYGIENE_CATEGORIES
    assert combined == set(SCORABLE_CATEGORIES), (
        "CONTENT | HYGIENE != SCORABLE\n"
        f"  only in SCORABLE: {set(SCORABLE_CATEGORIES) - combined}\n"
        f"  only in union: {combined - set(SCORABLE_CATEGORIES)}"
    )


def test_moat_categories_in_content():
    """The benchmark moat (prose_code_* + accession_id_prefix) must be CONTENT."""
    moat = {
        "prose_code_pkg_version", "prose_code_stat_test",
        "prose_code_marker", "prose_code_param",
        "accession_id_prefix",
    }
    assert moat.issubset(CONTENT_CATEGORIES), (
        f"moat categories missing from CONTENT: {moat - CONTENT_CATEGORIES}"
    )
