"""End-to-end tests for ``system_tests/generate_benchmark_heatmaps.py``.

The script consumes a ``STRESS_TEST_RESULTS.json`` and writes two PNGs.
Behaviour pinned here:

  * F1 and fix-rate are recomputed from ``category_breakdown`` so
    excluded categories don't contribute to either numerator or
    denominator.
  * The default exclusion set matches the "adjusted" reference figures
    (``code_func_name``, ``duplicate``, ``inline_code``,
    ``markdown_structure``).
  * Rows are grouped by strategy (simple → bioguider → pipeline),
    columns by error count; missing cells become NaN.
"""
from __future__ import annotations

import json
import math
import os
import sys
import tempfile

import matplotlib
matplotlib.use("Agg")

sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "system_tests")),
)
import generate_benchmark_heatmaps as gbh  # noqa: E402


# Minimal category partitions exercising the exclusion logic without
# depending on the real ``bioguider.managers.config`` import.
CONTENT_CATS = frozenset({"number", "param_name"})
HYGIENE_CATS = frozenset({"typo", "inline_code", "markdown_structure", "duplicate"})


def _make_row(model: str, error_count: int, breakdown: list[dict], precision: float = 1.0) -> dict:
    return {
        "model": model,
        "error_count": error_count,
        "precision_scorable": precision,
        "category_breakdown": breakdown,
    }


def _common_breakdown() -> list[dict]:
    """Five categories: two content, two hygiene-that-are-excluded, one
    hygiene-that-stays.  Lets each test reason about the inclusion bag."""
    return [
        {"category": "number",             "fixed": 8, "unfixed": 2},   # CONTENT, kept
        {"category": "param_name",         "fixed": 4, "unfixed": 1},   # CONTENT, kept
        {"category": "typo",               "fixed": 6, "unfixed": 4},   # HYGIENE, kept
        {"category": "inline_code",        "fixed": 0, "unfixed": 10},  # HYGIENE, EXCLUDED
        {"category": "markdown_structure", "fixed": 0, "unfixed": 10},  # HYGIENE, EXCLUDED
    ]


def test_default_excluded_set_matches_user_request():
    assert gbh.DEFAULT_EXCLUDED_CATEGORIES == frozenset({
        "code_func_name", "duplicate", "inline_code", "markdown_structure",
    })


def test_split_model_strategy_handles_plus_suffix():
    assert gbh._split_model_strategy("gpt-4o+pipeline") == ("gpt-4o", "pipeline")
    assert gbh._split_model_strategy("claude-sonnet-4-6+simple") == ("claude-sonnet-4-6", "simple")


def test_split_model_strategy_handles_missing_suffix():
    base, strat = gbh._split_model_strategy("solo")
    assert base == "solo" and strat == ""


def test_recompute_drops_excluded_from_both_numerator_and_denominator():
    """Without exclusion: 18 fixed / 45 injected = 0.40.  With the two
    hygiene-excluded rows stripped: 18 fixed / 25 injected = 0.72."""
    row = _make_row("gpt-4o+pipeline", 10, _common_breakdown())
    raw = gbh._recompute_row_metrics(row, frozenset(), CONTENT_CATS, HYGIENE_CATS)
    assert math.isclose(raw["fix_rate"], 18 / 45, abs_tol=1e-6)

    excluded = frozenset({"inline_code", "markdown_structure"})
    adj = gbh._recompute_row_metrics(row, excluded, CONTENT_CATS, HYGIENE_CATS)
    assert math.isclose(adj["fix_rate"], 18 / 25, abs_tol=1e-6)


def test_recompute_content_and_hygiene_f1_use_group_local_recall():
    row = _make_row("gpt-4o+pipeline", 10, _common_breakdown())
    out = gbh._recompute_row_metrics(
        row, gbh.DEFAULT_EXCLUDED_CATEGORIES, CONTENT_CATS, HYGIENE_CATS,
    )
    # CONTENT: fixed=12, injected=15, recall=12/15.  P=1.0 → F1=2*1*0.8/(1+0.8)
    expected_content = 2 * 1.0 * (12 / 15) / (1.0 + 12 / 15)
    assert math.isclose(out["f1_content"], expected_content, abs_tol=1e-6)
    # HYGIENE (after exclusion): only ``typo`` remains. fixed=6, injected=10.
    expected_hygiene = 2 * 1.0 * (6 / 10) / (1.0 + 6 / 10)
    assert math.isclose(out["f1_hygiene"], expected_hygiene, abs_tol=1e-6)


def test_recompute_returns_nan_when_a_group_is_fully_excluded():
    row = _make_row("gpt-4o+pipeline", 10, _common_breakdown())
    # Exclude every hygiene category: hygiene F1 should be NaN, not 0.0.
    excluded = frozenset({"typo", "inline_code", "markdown_structure", "duplicate"})
    out = gbh._recompute_row_metrics(row, excluded, CONTENT_CATS, HYGIENE_CATS)
    assert math.isnan(out["f1_hygiene"])
    assert not math.isnan(out["f1_content"])


def test_grid_row_order_is_strategy_grouped():
    results = [
        _make_row("gpt-4o+pipeline",  10, _common_breakdown()),
        _make_row("gpt-4o+simple",    10, _common_breakdown()),
        _make_row("gpt-4o+bioguider", 10, _common_breakdown()),
        _make_row("glm-5+pipeline",   10, _common_breakdown()),
        _make_row("glm-5+simple",     10, _common_breakdown()),
    ]
    rows, cols, _ = gbh._grid_from_breakdown(
        results, metric="fix_rate",
        excluded=gbh.DEFAULT_EXCLUDED_CATEGORIES,
        content_cats=CONTENT_CATS, hygiene_cats=HYGIENE_CATS,
    )
    seen = [gbh._split_model_strategy(r)[1] for r in rows]
    compacted = [s for i, s in enumerate(seen) if i == 0 or s != seen[i - 1]]
    assert compacted == ["simple", "bioguider", "pipeline"]
    assert cols == [10]


def test_grid_missing_cells_become_nan():
    results = [
        _make_row("gpt-4o+pipeline", 10, _common_breakdown()),
        _make_row("gpt-4o+pipeline", 40, _common_breakdown()),
        _make_row("glm-5+simple",    10, _common_breakdown()),
        # glm-5+simple has no error_count=40 → that cell must be NaN.
    ]
    rows, cols, matrix = gbh._grid_from_breakdown(
        results, metric="fix_rate",
        excluded=gbh.DEFAULT_EXCLUDED_CATEGORIES,
        content_cats=CONTENT_CATS, hygiene_cats=HYGIENE_CATS,
    )
    j40 = cols.index(40)
    for i, r in enumerate(rows):
        if r == "gpt-4o+pipeline":
            assert not math.isnan(matrix[i, j40])
        else:
            assert math.isnan(matrix[i, j40]), f"unexpected value at {r}"


def test_strategy_group_runs_partition_rows():
    rows = [
        "a+simple", "b+simple",
        "a+bioguider", "b+bioguider",
        "a+pipeline",
    ]
    runs = gbh._strategy_group_runs(rows)
    assert runs == [
        ("simple", 0, 1),
        ("bioguider", 2, 3),
        ("pipeline", 4, 4),
    ]


def _write_run(tmpdir: str, results: list[dict]) -> str:
    payload = {"timestamp": "2026-05-15", "unscorable_categories": [], "results": results}
    path = os.path.join(tmpdir, "STRESS_TEST_RESULTS.json")
    with open(path, "w") as fh:
        json.dump(payload, fh)
    return tmpdir


def test_generate_f1_heatmap_writes_png():
    with tempfile.TemporaryDirectory() as td:
        results = [
            _make_row("gpt-4o+pipeline", 10, _common_breakdown()),
            _make_row("gpt-4o+simple",   10, _common_breakdown()),
        ]
        _write_run(td, results)
        out = gbh.generate_f1_heatmap(
            td, results, content_cats=CONTENT_CATS, hygiene_cats=HYGIENE_CATS,
        )
        assert out.endswith("benchmark_f1_heatmap.png")
        assert os.path.exists(out) and os.path.getsize(out) > 1000


def test_generate_fix_rate_heatmap_writes_png():
    with tempfile.TemporaryDirectory() as td:
        results = [
            _make_row("gpt-4o+pipeline", 10, _common_breakdown()),
            _make_row("gpt-4o+simple",   10, _common_breakdown()),
        ]
        _write_run(td, results)
        out = gbh.generate_fix_rate_heatmap(
            td, results, content_cats=CONTENT_CATS, hygiene_cats=HYGIENE_CATS,
        )
        assert out.endswith("benchmark_fix_rate_heatmap.png")
        assert os.path.exists(out) and os.path.getsize(out) > 1000


def test_parse_argv_accepts_exclude_override():
    out = gbh._parse_argv(["script", "/some/path", "--exclude", "foo,bar"])
    assert out is not None
    run_dir, excluded = out
    assert run_dir == "/some/path"
    assert excluded == frozenset({"foo", "bar"})


def test_parse_argv_rejects_unknown_flag():
    assert gbh._parse_argv(["script", "/some/path", "--nope"]) is None


def test_parse_argv_uses_default_excluded_when_omitted():
    out = gbh._parse_argv(["script", "/some/path"])
    assert out is not None
    _run_dir, excluded = out
    assert excluded == gbh.DEFAULT_EXCLUDED_CATEGORIES


def test_main_prints_usage_on_wrong_argc(capsys):
    rc = gbh.main(["script"])
    assert rc == 2
    out = capsys.readouterr().out
    assert "STRESS_TEST_RESULTS.json" in out


def test_generate_per_level_emits_four_files_per_level():
    """For each error level, both adjusted and unadjusted F1 + fix-rate
    PNGs must land with the exact naming the user asked for."""
    with tempfile.TemporaryDirectory() as td:
        results = [
            _make_row("gpt-4o+pipeline", 10, _common_breakdown()),
            _make_row("gpt-4o+simple",   10, _common_breakdown()),
            _make_row("gpt-4o+pipeline", 40, _common_breakdown()),
            _make_row("gpt-4o+simple",   40, _common_breakdown()),
        ]
        written = gbh.generate_per_level_heatmaps(
            td, results, content_cats=CONTENT_CATS, hygiene_cats=HYGIENE_CATS,
        )
        for level in (10, 40):
            for suffix in ("", "_adjusted"):
                for metric in ("f1", "fix_rate"):
                    name = f"benchmark_{level}_{metric}_heatmap{suffix}.png"
                    assert os.path.join(td, name) in written, f"missing {name}"
                    assert os.path.exists(os.path.join(td, name))
                    assert os.path.getsize(os.path.join(td, name)) > 1000


def test_generate_per_level_skips_levels_with_no_rows():
    """Defensive: if a level has no data, no files should be emitted for it."""
    with tempfile.TemporaryDirectory() as td:
        results = [_make_row("gpt-4o+pipeline", 10, _common_breakdown())]
        written = gbh.generate_per_level_heatmaps(
            td, results, content_cats=CONTENT_CATS, hygiene_cats=HYGIENE_CATS,
        )
        # Exactly 4 files for the single level present.
        assert len(written) == 4
        for path in written:
            assert "benchmark_10_" in os.path.basename(path)


def test_filter_to_level_keeps_only_matching_rows():
    results = [
        _make_row("gpt-4o+pipeline", 10, _common_breakdown()),
        _make_row("gpt-4o+pipeline", 40, _common_breakdown()),
        _make_row("gpt-4o+simple",   10, _common_breakdown()),
    ]
    only10 = gbh._filter_to_level(results, 10)
    assert len(only10) == 2
    assert all(r["error_count"] == 10 for r in only10)
