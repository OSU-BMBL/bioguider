"""Regression test for CLI-category fix detection in ``BenchmarkEvaluator``.

The three CLI injection categories (``cli_flag_typo``, ``cli_unknown_flag``,
``cli_program_rename``) were initially unregistered in
``BenchmarkEvaluator._check_error_fixed``, causing every CLI mutation to
fall through to the default ``return False, "unchanged"`` branch — so even
when an LLM fixed the document, the evaluator never counted it.

These tests pin the corrected behaviour: when the mutated snippet has been
removed from the revised document, the error is reported as fixed; when the
mutated snippet is still present, it is reported as unchanged.
"""
import pytest

from bioguider.generation.benchmark_metrics import BenchmarkEvaluator


@pytest.fixture
def evaluator() -> BenchmarkEvaluator:
    return BenchmarkEvaluator(llm=None)


@pytest.mark.parametrize(
    "category, orig, mut, revised, expected_fixed",
    [
        # cli_flag_typo: --epochs truncated to --epoch
        (
            "cli_flag_typo",
            "python train.py --epochs 20",
            "python train.py --epoch 20",
            "python train.py --epochs 20",   # mut absent -> fixed
            True,
        ),
        (
            "cli_flag_typo",
            "python train.py --epochs 20",
            "python train.py --epoch 20",
            "python train.py --epoch 20",    # mut still present -> unchanged
            False,
        ),
        # cli_unknown_flag: bogus --workers appended
        (
            "cli_unknown_flag",
            "python train.py --epochs 20",
            "python train.py --epochs 20 --workers 4",
            "python train.py --epochs 20",   # mut absent -> fixed
            True,
        ),
        (
            "cli_unknown_flag",
            "python train.py --epochs 20",
            "python train.py --epochs 20 --workers 4",
            "python train.py --epochs 20 --workers 4",  # mut still present -> unchanged
            False,
        ),
        # cli_program_rename: pharokka_plotting -> pharokka_pltoting
        (
            "cli_program_rename",
            "pharokka_plotting.py -i x.tsv",
            "pharokka_pltoting.py -i x.tsv",
            "pharokka_plotting.py -i x.tsv",  # mut absent -> fixed
            True,
        ),
        (
            "cli_program_rename",
            "pharokka_plotting.py -i x.tsv",
            "pharokka_pltoting.py -i x.tsv",
            "pharokka_pltoting.py -i x.tsv",  # mut still present -> unchanged
            False,
        ),
    ],
)
def test_check_error_fixed_cli_categories(evaluator, category, orig, mut, revised, expected_fixed):
    baseline = orig  # baseline / corrupted aren't read by the CLI branch
    corrupted = mut
    is_fixed, status = evaluator._check_error_fixed(
        category=category,
        orig=orig,
        mut=mut,
        baseline=baseline,
        corrupted=corrupted,
        revised=revised,
    )
    assert is_fixed is expected_fixed, (category, status)
    if expected_fixed:
        assert status == "fixed_to_valid"
    else:
        assert status == "unchanged"


def test_evaluate_single_file_counts_cli_fix(evaluator):
    """End-to-end check via the public ``evaluate_single_file`` entry point."""
    baseline = "Run `python train.py --epochs 20` to train.\n"
    corrupted = "Run `python train.py --epoch 20` to train.\n"
    revised = baseline  # LLM restored the original
    manifest = {
        "errors": [
            {
                "id": "e1",
                "category": "cli_flag_typo",
                "original_snippet": "python train.py --epochs 20",
                "mutated_snippet": "python train.py --epoch 20",
            },
            {
                "id": "e2",
                "category": "cli_unknown_flag",
                "original_snippet": "python train.py --epochs 20",
                "mutated_snippet": "python train.py --epoch 20 --workers 4",  # not in revised
            },
        ]
    }
    metrics, _ = evaluator.evaluate_single_file(
        baseline=baseline,
        corrupted=corrupted,
        revised=revised,
        injection_manifest=manifest,
        file_path="train_docs.md",
        file_category="userguide",
        detect_semantic_fp=False,
    )
    assert [m.is_fixed for m in metrics] == [True, True]
    assert [m.status for m in metrics] == ["fixed_to_valid", "fixed_to_valid"]


# ── Regression: token-anchored CLI fix check ─────────────────────────────


def test_cli_unknown_flag_false_fix_from_whitespace_normalisation(evaluator):
    """The pharokka run uncovered this: the injector preserves the doc's
    double-space ``pharokka.gbk  -o`` formatting in the mutated line, but
    the LLM normalises whitespace when re-emitting the doc. The whole-line
    ``mut not in revised`` check then falsely reports the bug as fixed.
    With ``mutated_token`` populated, the evaluator looks for ``--cores``
    directly — which is still in the revised file — and correctly reports
    it unfixed.
    """
    orig = "pharokka_multiplotter.py -g pharokka.gbk  -o out/"
    mut = "pharokka_multiplotter.py -g pharokka.gbk  -o out/ --cores 4"
    # LLM collapsed the double-space, but `--cores 4` is still very much present.
    revised = "pharokka_multiplotter.py -g pharokka.gbk -o out/ --cores 4\n"
    manifest = {
        "errors": [
            {
                "id": "e1",
                "category": "cli_unknown_flag",
                "original_snippet": orig,
                "mutated_snippet": mut,
                "mutated_token": "--cores",
            }
        ]
    }
    metrics, _ = evaluator.evaluate_single_file(
        baseline=orig,
        corrupted=mut,
        revised=revised,
        injection_manifest=manifest,
        file_path="plotting.md",
        file_category="userguide",
        detect_semantic_fp=False,
    )
    assert metrics[0].is_fixed is False
    assert metrics[0].status == "unchanged"


def test_cli_unknown_flag_token_check_reports_fixed_when_token_gone(evaluator):
    """Mirror case: token is gone from the revised file → fixed."""
    orig = "pharokka_multiplotter.py -g pharokka.gbk -o out/"
    mut = "pharokka_multiplotter.py -g pharokka.gbk -o out/ --cores 4"
    revised = orig + "\n"
    manifest = {
        "errors": [
            {
                "id": "e1",
                "category": "cli_unknown_flag",
                "original_snippet": orig,
                "mutated_snippet": mut,
                "mutated_token": "--cores",
            }
        ]
    }
    metrics, _ = evaluator.evaluate_single_file(
        baseline=orig,
        corrupted=mut,
        revised=revised,
        injection_manifest=manifest,
        file_path="plotting.md",
        file_category="userguide",
        detect_semantic_fp=False,
    )
    assert metrics[0].is_fixed is True
    assert metrics[0].status == "fixed_to_valid"


def test_cli_flag_typo_token_anchored_match_avoids_substring_collision(evaluator):
    """The truncated form ``--epoch`` must NOT match the full ``--epochs``
    in a revised file (``--epoch`` is a substring of ``--epochs``). The
    token check uses (?<!\\S)…(?!\\S) anchors so the match is standalone."""
    revised = "python train.py --epochs 20\n"   # LLM restored full flag
    manifest = {
        "errors": [
            {
                "id": "e1",
                "category": "cli_flag_typo",
                "original_snippet": "python train.py --epochs 20",
                "mutated_snippet": "python train.py --epoch 20",
                "mutated_token": "--epoch",
            }
        ]
    }
    metrics, _ = evaluator.evaluate_single_file(
        baseline=revised,
        corrupted="python train.py --epoch 20\n",
        revised=revised,
        injection_manifest=manifest,
        file_path="d.md",
        file_category="userguide",
        detect_semantic_fp=False,
    )
    # `--epoch` is NOT a standalone token in revised (which has --epochs),
    # so the evaluator correctly sees it as fixed.
    assert metrics[0].is_fixed is True
    assert metrics[0].status == "fixed_to_valid"


def test_cli_legacy_manifest_without_mutated_token_uses_whole_line_fallback(evaluator):
    """Manifests written before ``mutated_token`` was added should keep the
    old whole-line behaviour (no schema migration needed)."""
    orig = "python train.py --epochs 20"
    mut = "python train.py --epoch 20"
    revised = orig + "\n"   # whole line restored
    manifest = {
        "errors": [
            {
                "id": "e1",
                "category": "cli_flag_typo",
                "original_snippet": orig,
                "mutated_snippet": mut,
                # mutated_token deliberately absent
            }
        ]
    }
    metrics, _ = evaluator.evaluate_single_file(
        baseline=orig,
        corrupted=mut,
        revised=revised,
        injection_manifest=manifest,
        file_path="d.md",
        file_category="userguide",
        detect_semantic_fp=False,
    )
    assert metrics[0].is_fixed is True
    assert metrics[0].status == "fixed_to_valid"
