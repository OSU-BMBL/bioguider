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
        assert set(d["unscorable_categories"]) == {
            "function", "comment_typo", "code_lang_tag", "code_func_name", "code_func_args"
        }

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


class TestBenchmarkResultScorable:
    """D2-style scorable carve-out in BenchmarkResult (stress-test path)."""

    def _result_with(self, function_tp=0, function_fn=0, typo_tp=0, typo_fn=0, fp=0):
        from bioguider.generation.benchmark_metrics import (
            BenchmarkResult,
            ErrorMetrics,
        )

        errors = []
        for _ in range(function_tp):
            errors.append(ErrorMetrics("e", "function", "f", True, "", "", "fixed_to_baseline"))
        for _ in range(function_fn):
            errors.append(ErrorMetrics("e", "function", "f", False, "", "", "unchanged"))
        for _ in range(typo_tp):
            errors.append(ErrorMetrics("e", "typo", "f", True, "", "", "fixed_to_baseline"))
        for _ in range(typo_fn):
            errors.append(ErrorMetrics("e", "typo", "f", False, "", "", "unchanged"))
        r = BenchmarkResult(
            error_count=10,
            file_count=1,
            true_positives=function_tp + typo_tp,
            false_negatives=function_fn + typo_fn,
            false_positives=fp,
            error_details=errors,
        )
        r.compute_derived_metrics()
        return r

    def test_scorable_excludes_function(self):
        r = self._result_with(function_tp=5, typo_tp=3, typo_fn=2)
        # Headline sees 8 / 10 (80%). Scorable sees 3 / 5 (60%).
        assert r.fix_rate == 0.8
        assert r.fix_rate_scorable == 0.6
        assert r.f1_score_scorable < r.f1_score

    def test_no_unscorable_yields_parity(self):
        r = self._result_with(typo_tp=4, typo_fn=1)
        assert r.f1_score == r.f1_score_scorable

    def test_to_dict_has_scorable_keys(self):
        r = self._result_with(typo_tp=1, function_tp=1)
        d = r.to_dict()
        for k in (
            "f1_score_scorable",
            "precision_scorable",
            "recall_scorable",
            "fix_rate_scorable",
            "true_positives_scorable",
            "false_negatives_scorable",
            "total_errors_scorable",
            "unscorable_categories",
        ):
            assert k in d, f"Missing scorable key in BenchmarkResult.to_dict: {k}"
        assert set(d["unscorable_categories"]) == {
            "function", "comment_typo", "code_lang_tag", "code_func_name", "code_func_args"
        }


class TestSharedHelperParity:
    """The shared ``compute_scorable_breakdown`` produces identical numbers
    whether fed dataclass instances or plain dicts — both evaluators rely on this."""

    def test_dict_and_dataclass_parity(self):
        from bioguider.generation.benchmark_metrics import ErrorMetrics
        from bioguider.managers.config import compute_scorable_breakdown

        dc = [
            ErrorMetrics("a", "typo", "f", True, "", "", "fixed_to_baseline"),
            ErrorMetrics("b", "typo", "f", False, "", "", "unchanged"),
            ErrorMetrics("c", "function", "f", True, "", "", "fixed_to_baseline"),
        ]
        dicts = [
            {"category": "typo", "is_fixed": True},
            {"category": "typo", "is_fixed": False},
            {"category": "function", "is_fixed": True},
        ]
        a = compute_scorable_breakdown(dc, 0)
        b = compute_scorable_breakdown(dicts, 0)
        assert a == b
