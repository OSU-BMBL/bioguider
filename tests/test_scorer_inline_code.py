"""Regression tests for the inline_code scorer substring-containment bug.

Before the T1 fix, the scorer's predicate was:
    rewrapped in fixed AND mut not in fixed

which is tautologically false because ``rewrapped`` is literally
``f"`{raw}`"`` and always contains ``mut`` as a substring when ``mut``
is the same raw token (with or without the backticks).

To confirm this suite catches the bug: stash the T1 patch
(`git stash push bioguider/generation/unified_metrics.py
bioguider/generation/benchmark_metrics.py
system_tests/test_single_file_stress.py`), run
``pytest tests/test_scorer_inline_code.py``, and observe failures.
Then ``git stash pop`` to restore the fix.
"""
from __future__ import annotations

import pytest

from bioguider.generation.unified_metrics import ErrorChecker, _naked_count


def _score_via_unified(orig, mut, corrupted, fixed):
    ec = ErrorChecker()
    is_fixed, _status = ec._check_inline_code(orig, mut, fixed, corrupted, fixed)
    return is_fixed


def _score_via_benchmark_metrics(orig, mut, corrupted, fixed):
    raw = mut.strip("`") if mut else ""
    if not raw:
        return False
    return _naked_count(fixed, raw) < _naked_count(corrupted, raw)


def _score_via_stress_test(orig, mut, corrupted, fixed):
    raw = mut.strip("`") if mut else ""
    if not raw:
        return False
    return _naked_count(fixed, raw) < _naked_count(corrupted, raw)


SCORERS = [
    ("unified_metrics", _score_via_unified),
    ("benchmark_metrics", _score_via_benchmark_metrics),
    ("stress_test", _score_via_stress_test),
]


@pytest.mark.parametrize("name,scorer", SCORERS)
def test_inline_code_rewrapped_is_fixed(name, scorer):
    """Model correctly restored backticks => is_fixed True."""
    corrupted = "Call FindMarker() to run the test."
    fixed = "Call `FindMarker()` to run the test."
    assert scorer("`FindMarker()`", "FindMarker()", corrupted, fixed), (
        f"{name}: rewrapped form must score as fixed"
    )


@pytest.mark.parametrize("name,scorer", SCORERS)
def test_inline_code_noop_not_fixed(name, scorer):
    """Model did nothing => is_fixed False."""
    corrupted = "Call FindMarker() to run the test."
    fixed = corrupted
    assert not scorer("`FindMarker()`", "FindMarker()", corrupted, fixed), (
        f"{name}: no-op must not score as fixed"
    )


@pytest.mark.parametrize("name,scorer", SCORERS)
def test_inline_code_removal_counts_as_fixed(name, scorer):
    """Model removed the naked form entirely (by rephrasing) => is_fixed True (net decrease)."""
    corrupted = "Call FindMarker() to run the test."
    fixed = "Run the standard Seurat differential expression analysis."
    assert scorer("`FindMarker()`", "FindMarker()", corrupted, fixed), (
        f"{name}: removal via rephrase must score as fixed"
    )
