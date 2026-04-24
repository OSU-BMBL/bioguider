"""Tests for D2 — UNSCORABLE_CATEGORIES filter in EvaluationResult.

Rationale: the `function`-name category is structurally unfixable by BioGuider
because the locator uses function names as anchors for doc context. These
errors are injected for realism but must not contribute to headline F1.
"""

from bioguider.generation.unified_metrics import (
    ErrorEvaluation,
    EvaluationResult,
    FixStatus,
)
from bioguider.managers.config import (
    ALL_ERROR_CATEGORIES,
    SCORABLE_CATEGORIES,
    UNSCORABLE_CATEGORIES,
)


def _ev(cat: str, is_fixed: bool, idx: int = 0) -> ErrorEvaluation:
    return ErrorEvaluation(
        error_id=f"e{idx}",
        category=cat,
        file_path="test.md",
        status=FixStatus.FIXED_TO_BASELINE if is_fixed else FixStatus.UNCHANGED,
        is_fixed=is_fixed,
    )


class TestConfigShape:
    def test_function_in_unscorable(self):
        assert "function" in UNSCORABLE_CATEGORIES

    def test_scorable_plus_unscorable_equals_all(self):
        assert SCORABLE_CATEGORIES | UNSCORABLE_CATEGORIES == ALL_ERROR_CATEGORIES
        assert SCORABLE_CATEGORIES & UNSCORABLE_CATEGORIES == frozenset()

    def test_scorable_categories_nonempty(self):
        assert len(SCORABLE_CATEGORIES) > 30


class TestScorableMetricsSplit:
    """AC: f1_scorable excludes UNSCORABLE_CATEGORIES; f1 (headline) includes everything."""

    def test_headline_counts_unscorable_scorable_excludes(self):
        # 3 scorable fixed, 2 scorable unfixed, 5 unscorable fixed, 0 unscorable unfixed
        evs = (
            [_ev("typo", True, i) for i in range(3)]
            + [_ev("typo", False, i + 10) for i in range(2)]
            + [_ev("function", True, i + 20) for i in range(5)]
        )
        result = EvaluationResult(
            true_positives=8,  # 3 + 5
            false_negatives=2,
            false_positives=0,
            error_evaluations=evs,
        )
        result.compute_metrics()

        # Headline F1 sees all 10 errors.
        assert result.true_positives == 8
        assert result.false_negatives == 2
        # Scorable F1 sees only the 5 typo errors (3 fixed, 2 unfixed).
        assert result.true_positives_scorable == 3
        assert result.false_negatives_scorable == 2
        assert result.total_errors_scorable == 5

        # f1_score includes the "free" function fixes (fix rate 80%).
        # f1_score_scorable is 60% because typo errors are 3/5.
        assert result.fix_rate == 0.8
        assert result.fix_rate_scorable == 0.6
        assert result.f1_score > result.f1_score_scorable

    def test_no_unscorable_present_metrics_match(self):
        evs = [_ev("typo", True, i) for i in range(4)] + [_ev("typo", False, 99)]
        result = EvaluationResult(
            true_positives=4,
            false_negatives=1,
            false_positives=0,
            error_evaluations=evs,
        )
        result.compute_metrics()
        assert result.f1_score == result.f1_score_scorable
        assert result.precision_scorable == result.precision
        assert result.recall_scorable == result.recall

    def test_all_unscorable_scorable_metrics_zero(self):
        evs = [_ev("function", True, i) for i in range(3)]
        result = EvaluationResult(
            true_positives=3,
            false_negatives=0,
            false_positives=0,
            error_evaluations=evs,
        )
        result.compute_metrics()
        assert result.total_errors_scorable == 0
        assert result.f1_score_scorable == 0.0
        assert result.precision_scorable == 0.0
        assert result.recall_scorable == 0.0

    def test_to_dict_exposes_scorable_fields(self):
        evs = [_ev("typo", True, 1), _ev("function", False, 2)]
        result = EvaluationResult(
            true_positives=1,
            false_negatives=1,
            false_positives=0,
            error_evaluations=evs,
        )
        result.compute_metrics()
        d = result.to_dict()
        for key in (
            "f1_score_scorable",
            "precision_scorable",
            "recall_scorable",
            "fix_rate_scorable",
            "true_positives_scorable",
            "false_negatives_scorable",
            "false_positives_scorable",
            "total_errors_scorable",
            "unscorable_categories",
        ):
            assert key in d, f"Missing in to_dict(): {key}"
        assert d["unscorable_categories"] == ["function"]

    def test_fp_mirror_into_scorable(self):
        evs = [_ev("typo", True, 1)]
        result = EvaluationResult(
            true_positives=1,
            false_negatives=0,
            false_positives=4,
            error_evaluations=evs,
        )
        result.compute_metrics()
        # FPs are category-agnostic so they mirror to the scorable bucket as-is.
        assert result.false_positives_scorable == 4
        assert result.precision_scorable == 1 / 5
