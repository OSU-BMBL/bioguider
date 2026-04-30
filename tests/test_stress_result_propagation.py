"""
Regression tests for StressLevelResult -> JSON/CSV propagation.

Caught a bug where new fields (false_positives, code_fence_violations,
yaml_violations, section_violations) were added to BenchmarkResult and
StressLevelResult but never serialized, leaving smoke-test gates unable
to verify they were populated.
"""
import csv
import json
import os
import tempfile

from system_tests.test_single_file_stress import (
    CategoryResult,
    StressLevelResult,
    save_results,
)


def _make_result(**overrides):
    """Build a minimally-populated StressLevelResult with sensible defaults."""
    base = dict(
        error_count=10,
        total_errors_injected=100,
        errors_fixed=80,
        errors_unfixed=20,
        fix_rate=0.8,
        precision=0.95,
        recall=0.8,
        f1_score=0.87,
        duration_seconds=5.0,
        category_results=[
            CategoryResult(category="typo", injected=10, fixed=8, unfixed=2, fix_rate=0.8),
            CategoryResult(category="link", injected=10, fixed=7, unfixed=3, fix_rate=0.7),
        ],
        model_name="gpt-4o+bioguider",
    )
    base.update(overrides)
    return StressLevelResult(**base)


def test_protection_fields_serialize_to_json():
    """code_fence_violations, yaml_violations, section_violations, false_positives must appear in JSON."""
    r = _make_result(
        false_positives=4,
        code_fence_violations=2,
        yaml_violations=1,
        section_violations=3,
    )
    with tempfile.TemporaryDirectory() as td:
        save_results([r], td)
        with open(os.path.join(td, "STRESS_TEST_RESULTS.json")) as f:
            data = json.load(f)

    row = data["results"][0]
    assert row["false_positives"] == 4
    assert row["code_fence_violations"] == 2
    assert row["yaml_violations"] == 1
    assert row["section_violations"] == 3


def test_protection_fields_serialize_to_csv():
    """Same fields must appear in STRESS_TEST_TABLE.csv."""
    r = _make_result(
        false_positives=7,
        code_fence_violations=5,
        yaml_violations=0,
        section_violations=2,
    )
    with tempfile.TemporaryDirectory() as td:
        save_results([r], td)
        with open(os.path.join(td, "STRESS_TEST_TABLE.csv")) as f:
            rows = list(csv.DictReader(f))

    assert len(rows) == 1
    row = rows[0]
    assert int(row["false_positives"]) == 7
    assert int(row["code_fence_violations"]) == 5
    assert int(row["yaml_violations"]) == 0
    assert int(row["section_violations"]) == 2


def test_protection_fields_default_zero():
    """Protection fields default to 0 when not set, so legacy callers still work."""
    r = _make_result()  # No protection fields passed

    assert r.false_positives == 0
    assert r.code_fence_violations == 0
    assert r.yaml_violations == 0
    assert r.section_violations == 0

    with tempfile.TemporaryDirectory() as td:
        save_results([r], td)
        with open(os.path.join(td, "STRESS_TEST_RESULTS.json")) as f:
            data = json.load(f)

    row = data["results"][0]
    assert row["false_positives"] == 0
    assert row["code_fence_violations"] == 0
    assert row["yaml_violations"] == 0
    assert row["section_violations"] == 0


def test_skill_csv_writer_includes_protection_columns():
    """SKILL_COMPARISON.csv and SKILL_MATRIX_TABLE.csv must include protection columns."""
    from system_tests.test_single_file_stress import _write_skill_comparison_csv

    rows = [{
        "file_stem": "demo_vignette",
        "model": "gpt-4o",
        "skill": "bioguider",
        "error_count": 30,
        "total_injected": 200,
        "fixed": 160,
        "unfixed": 40,
        "fix_rate": 0.8,
        "f1_score": 0.85,
        "f1_score_scorable": 0.83,
        "f1_score_content": 0.92,
        "f1_score_hygiene": 0.74,
        "false_positives": 5,
        "code_fence_violations": 1,
        "yaml_violations": 0,
        "section_violations": 2,
        "duration_s": 45.2,
    }]
    with tempfile.TemporaryDirectory() as td:
        csv_path = os.path.join(td, "SKILL_COMPARISON.csv")
        _write_skill_comparison_csv(rows, csv_path)
        with open(csv_path) as f:
            out = list(csv.DictReader(f))

    assert len(out) == 1
    row = out[0]
    assert "false_positives" in row and int(row["false_positives"]) == 5
    assert "code_fence_violations" in row and int(row["code_fence_violations"]) == 1
    assert "yaml_violations" in row and int(row["yaml_violations"]) == 0
    assert "section_violations" in row and int(row["section_violations"]) == 2


def test_precision_scorable_uses_actual_false_positives():
    """precision_scorable must reflect false_positives, not always 1.0.

    Catches the regression where _populate_scorable read r.false_positives
    (default 0) and computed precision_scorable = TP/(TP+0) = 1.0 even
    when BenchmarkResult had detected collateral-damage FPs.
    """
    r = _make_result(false_positives=10)  # 10 FPs against 80 TPs
    with tempfile.TemporaryDirectory() as td:
        save_results([r], td)
        with open(os.path.join(td, "STRESS_TEST_RESULTS.json")) as f:
            data = json.load(f)

    row = data["results"][0]
    # category_results sum: 8+7 = 15 TPs scorable, 5 FNs scorable
    # precision_scorable = 15 / (15 + 10) = 0.6
    assert row["precision_scorable"] < 1.0
    assert row["precision_scorable"] == 0.6
