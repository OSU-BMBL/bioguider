"""Tests for D3 — total-error budget knob for the F1-vs-error-count gradient.

Covers:
- TOTAL_ERROR_LEVELS exposes the 50/100/200/300 gradient points.
- ``min_per_category_from_total`` translates the budget correctly and always
  returns at least 1 (so every eligible slot still fires).
- ``inject_errors_parallel`` and ``run_total_error_gradient`` accept the new
  parameter and helper method signatures.
"""

import inspect

from bioguider.managers.benchmark_manager import BenchmarkManager
from bioguider.managers.config import (
    SCORABLE_CATEGORIES,
    TOTAL_ERROR_LEVELS,
    min_per_category_from_total,
)


class TestConfigConstants:
    def test_total_error_levels_values(self):
        assert TOTAL_ERROR_LEVELS == [50, 100, 200, 300]

    def test_scorable_categories_nonempty(self):
        assert len(SCORABLE_CATEGORIES) >= 30


class TestBudgetTranslation:
    """AC: budget math is (ceil-divided, floor-clamped) across files × categories."""

    def test_even_spread_simple(self):
        # 300 errors / (10 files × 30 categories) = 1.0 → ceil → 1
        assert min_per_category_from_total(300, n_files=10, n_categories=30) == 1

    def test_ceil_rounding_rounds_up(self):
        # 50 / (2 × 3) = 8.33 → ceil → 9
        assert min_per_category_from_total(50, n_files=2, n_categories=3) == 9

    def test_floor_one_prevents_zero(self):
        # 10 / (20 × 30) = 0.016 → ceil → 1 (must not be 0)
        assert min_per_category_from_total(10, n_files=20, n_categories=30) == 1

    def test_zero_total_still_returns_one(self):
        # Defensive: degenerate input still fires something.
        assert min_per_category_from_total(0, n_files=5, n_categories=5) == 1

    def test_zero_files_defensive(self):
        # Shouldn't divide by zero even if select_target_files returned none.
        val = min_per_category_from_total(100, n_files=0, n_categories=10)
        assert val >= 1


class TestMethodSignatures:
    """AC: the new knob and helper exist with the documented signatures."""

    def test_inject_errors_parallel_accepts_target_total_errors(self):
        sig = inspect.signature(BenchmarkManager.inject_errors_parallel)
        assert "target_total_errors" in sig.parameters
        param = sig.parameters["target_total_errors"]
        assert param.default is None

    def test_run_total_error_gradient_exists(self):
        assert callable(getattr(BenchmarkManager, "run_total_error_gradient"))
        sig = inspect.signature(BenchmarkManager.run_total_error_gradient)
        # First two required params (plus self).
        names = list(sig.parameters.keys())
        assert names[0] == "self"
        assert "report_path" in names
        assert "baseline_repo_path" in names
        assert "output_base_path" in names
        assert "total_levels" in names
        assert sig.parameters["total_levels"].default is None
