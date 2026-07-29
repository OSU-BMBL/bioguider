"""Regression tests for Pattern A installation/readme failures.

1. Empty installation-file list must NOT raise
   ``'NoneType' object is not subscriptable`` (it used to drop the whole page).
2. ``read_file`` must tolerate non-UTF-8 bytes instead of raising UnicodeDecodeError.
"""
import pytest

from bioguider.agents.evaluation_installation_task import EvaluationInstallationTask
from bioguider.agents.agent_utils import read_file
from bioguider.utils.constants import EvaluationInstallationResult


def test_evaluate_with_no_install_files_returns_absent_result():
    # llm=None is safe: the empty file list short-circuits before any LLM call.
    task = EvaluationInstallationTask(
        llm=None, repo_path="/tmp", gitignore_path="/tmp/.gitignore",
    )
    result, token_usage, files = task._evaluate([])

    assert isinstance(result, EvaluationInstallationResult)
    assert files == []
    # downstream page generator dereferences these, so they must not be None
    assert result.structured_evaluation is not None
    assert result.free_evaluation is not None
    assert result.structured_evaluation.install_available is False
    assert result.structured_evaluation.overall_score == 0
    assert result.free_evaluation.installation_guide  # non-empty message


def test_read_file_tolerates_non_utf8_bytes(tmp_path):
    p = tmp_path / "README.md"
    p.write_bytes(b"# Title\n\xf3 stray non-utf8 byte\nstill readable")

    content = read_file(p)

    assert content is not None
    assert "Title" in content and "still readable" in content
    assert "�" in content  # the bad byte became the replacement char


# ---- Pattern C: installation overall_score must not be a fake 0 ----

def test_overall_score_floors_for_documented_dependencies():
    # all four yes/no checks fail but a dependency manifest is documented:
    # used to collapse to 0, now floored to 20 to match the free-text rating
    score = EvaluationInstallationTask._compute_overall_score(
        False, False, False, False, dependency_number=15,
    )
    assert score == 20


def test_overall_score_genuine_zero_stays_zero():
    # nothing documented at all (no deps either) -> a true 0
    score = EvaluationInstallationTask._compute_overall_score(
        False, False, False, False, dependency_number=0,
    )
    assert score == 0


def test_overall_score_unaffected_when_other_signal_present():
    # good install keeps full score; install-only keeps its existing value
    assert EvaluationInstallationTask._compute_overall_score(True, True, True, True, 12) == 100
    assert EvaluationInstallationTask._compute_overall_score(True, False, False, False, 0) == 38
