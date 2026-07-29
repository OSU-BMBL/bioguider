"""
Unit tests for the inline_code_mismatch error category.

The injector moves the backticks off a genuine code token onto an adjacent
stopword, so the markup wraps the WRONG token (`` `--gff` and `` -> ``--gff `and` ``).
Scoring is per-error via the recorded mutated_token (the spurious `` `word` ``).
All tests are deterministic (no LLM / network).
"""
import pytest

from bioguider.generation.llm_injector import LLMErrorInjector
from bioguider.generation.benchmark_metrics import BenchmarkEvaluator
from bioguider.managers.config import (
    ALL_ERROR_CATEGORIES,
    SCORABLE_CATEGORIES,
    HYGIENE_CATEGORIES,
)


DOC = """\
# Tool

Run the `--gff` and check the output.
Pass the `--prefix` then set a name.
The `pharokka.py` with default options.
Use the `--threads` for speed.

```bash
pharokka.py -i input.fasta -o out
```
"""


def _make_injector():
    return LLMErrorInjector(llm=None, force_deterministic=True)


def _errors(manifest):
    return manifest.get("errors", [])


# ---------------------------------------------------------------------------
# Taxonomy registration
# ---------------------------------------------------------------------------

def test_category_registered_and_scorable():
    assert "inline_code_mismatch" in ALL_ERROR_CATEGORIES
    assert "inline_code_mismatch" in SCORABLE_CATEGORIES
    assert "inline_code_mismatch" in HYGIENE_CATEGORIES


# ---------------------------------------------------------------------------
# Injection
# ---------------------------------------------------------------------------

def test_injects_inline_code_mismatch():
    inj = _make_injector()
    corrupted, manifest = inj.inject(DOC, min_per_category=1)
    icm = [e for e in _errors(manifest) if e["category"] == "inline_code_mismatch"]
    assert icm, "expected at least one inline_code_mismatch error"


def test_mutation_moves_backticks_to_stopword():
    inj = _make_injector()
    corrupted, manifest = inj.inject(DOC, min_per_category=1)
    icm = [e for e in _errors(manifest) if e["category"] == "inline_code_mismatch"]
    e = icm[0]
    # original wraps a code-like token; mutated wraps a plain stopword.
    assert e["original_snippet"] != e["mutated_snippet"]
    assert e["mutated_token"].startswith("`") and e["mutated_token"].endswith("`")
    inner = e["mutated_token"].strip("`")
    assert inner.isalpha()  # a plain word, not a code token
    # the spurious wrapping is actually present in the corrupted doc
    assert e["mutated_token"] in corrupted
    # ...and the original (correct) wrapping is gone
    assert e["original_snippet"] not in corrupted


def test_does_not_break_code_fences():
    inj = _make_injector()
    corrupted, manifest = inj.inject(DOC, min_per_category=1)
    assert corrupted.count("```") == DOC.count("```")


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _check(orig, mut, mutated_token, corrupted, revised):
    ev = BenchmarkEvaluator(llm=None)
    return ev._check_error_fixed(
        "inline_code_mismatch", orig, mut, baseline="", corrupted=corrupted,
        revised=revised, mutated_token=mutated_token,
    )


def test_scoring_fixed_when_original_restored():
    orig, mut, tok = "`--gff` and", "--gff `and`", "`and`"
    fixed, status = _check(orig, mut, tok, corrupted=mut, revised=orig)
    assert fixed and status == "fixed_to_baseline"


def test_scoring_fixed_when_spurious_wrapping_removed():
    orig, mut, tok = "`--gff` and", "--gff `and`", "`and`"
    # model un-wrapped the word differently (e.g. dropped both backticks)
    revised = "--gff and"
    fixed, status = _check(orig, mut, tok, corrupted=mut, revised=revised)
    assert fixed and status == "fixed_to_valid"


def test_scoring_unfixed_when_mutation_remains():
    orig, mut, tok = "`--gff` and", "--gff `and`", "`and`"
    fixed, status = _check(orig, mut, tok, corrupted=mut, revised=mut)
    assert not fixed and status == "unchanged"
