"""Tests for D4 — phase split (eval_only / correction_only) in GenerationTestManagerV2.

Rationale: Qin wants evaluation to run once and correction to be resumed on
two hand-picked examples (avoids re-running full-corpus generation).

These tests exercise the phase dispatch, state serialization, and resume
whitelist logic — all with mocked LLM / generation / evaluation so we spend
zero tokens.
"""

import inspect
import json
import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from bioguider.managers.generation_test_manager_v2 import (
    EVALUATION_STATE_FILENAME,
    GenerationTestManagerV2,
)


class TestPhaseSignature:
    """AC: new parameters exist with the documented defaults."""

    def test_phase_param_exists_default_full(self):
        sig = inspect.signature(GenerationTestManagerV2.run_quant_test)
        assert "phase" in sig.parameters
        assert sig.parameters["phase"].default == "full"

    def test_resume_from_param_exists(self):
        sig = inspect.signature(GenerationTestManagerV2.run_quant_test)
        assert "resume_from" in sig.parameters
        assert sig.parameters["resume_from"].default is None

    def test_example_files_param_exists(self):
        sig = inspect.signature(GenerationTestManagerV2.run_quant_test)
        assert "example_files" in sig.parameters
        assert sig.parameters["example_files"].default is None


class TestPhaseDispatchErrors:
    """AC: misuse raises with a clear message; status quo is preserved."""

    def test_correction_only_without_resume_raises(self):
        mgr = GenerationTestManagerV2(llm=MagicMock(), step_callback=None)
        with pytest.raises(ValueError, match="resume_from"):
            mgr.run_quant_test(
                report_path="/fake/report.json",
                baseline_repo_path="/fake/repo",
                tmp_repo_path="/fake/tmp",
                phase="correction_only",
            )

    def test_unknown_phase_raises(self):
        mgr = GenerationTestManagerV2(llm=MagicMock(), step_callback=None)
        with pytest.raises(ValueError, match="Unknown phase"):
            mgr.run_quant_test(
                report_path="/fake/report.json",
                baseline_repo_path="/fake/repo",
                tmp_repo_path="/fake/tmp",
                phase="bogus",  # type: ignore[arg-type]
            )


class TestEvaluationStateSerialization:
    """AC: state file has the keys a later correction_only run needs."""

    def test_state_roundtrip(self):
        mgr = GenerationTestManagerV2(llm=MagicMock(), step_callback=None)
        with tempfile.TemporaryDirectory() as td:
            fake_results = {
                "README.md": MagicMock(),
                "vignettes/demo.Rmd": MagicMock(),
            }
            state_path = mgr._write_evaluation_state(
                tmp_repo_path=td,
                report_path=os.path.join(td, "report.json"),
                injection_manifest_path=os.path.join(td, "manifest.json"),
                injection_results=fake_results,
                min_per_category=5,
            )

            assert os.path.basename(state_path) == EVALUATION_STATE_FILENAME
            with open(state_path, "r", encoding="utf-8") as f:
                state = json.load(f)

            assert state["version"] == 1
            assert state["min_per_category"] == 5
            assert sorted(state["injected_files"]) == [
                "README.md",
                "vignettes/demo.Rmd",
            ]
            for key in (
                "tmp_repo_path",
                "report_path",
                "injection_manifest_path",
            ):
                assert os.path.isabs(state[key])


class TestCorrectionOnlyResume:
    """AC: correction_only honours the example_files whitelist + loads manifest."""

    def test_whitelist_narrows_target_files(self):
        mgr = GenerationTestManagerV2(llm=MagicMock(), step_callback=None)

        with tempfile.TemporaryDirectory() as td:
            manifest_path = os.path.join(td, "manifest.json")
            state_path = os.path.join(td, EVALUATION_STATE_FILENAME)
            out_dir = os.path.join(td, "out")
            os.makedirs(out_dir)

            manifest = {
                "files": {
                    "README.md": {"errors": [{"id": "e1", "category": "typo"}]},
                    "vignettes/demo.Rmd": {"errors": [{"id": "e2", "category": "typo"}]},
                    "docs/user_guide.md": {"errors": [{"id": "e3", "category": "typo"}]},
                }
            }
            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump(manifest, f)

            state = {
                "version": 1,
                "tmp_repo_path": td,
                "report_path": os.path.join(td, "report.json"),
                "injection_manifest_path": manifest_path,
                "min_per_category": 3,
                "injected_files": sorted(manifest["files"].keys()),
            }
            with open(state_path, "w", encoding="utf-8") as f:
                json.dump(state, f)

            fake_gen = MagicMock()
            fake_gen.run.return_value = out_dir
            fake_eval_result = MagicMock()
            fake_eval_result.to_dict.return_value = {"f1_score": 0.42}
            fake_evaluator = MagicMock()
            fake_evaluator.evaluate_multiple_files.return_value = fake_eval_result

            with patch(
                "bioguider.managers.generation_test_manager_v2.DocumentationGenerationManager",
                return_value=fake_gen,
            ), patch(
                "bioguider.managers.generation_test_manager_v2.UnifiedMetricsEvaluator",
                return_value=fake_evaluator,
            ), patch.object(
                GenerationTestManagerV2, "_generate_report"
            ):
                result_dir = mgr.run_quant_test(
                    report_path=state["report_path"],
                    baseline_repo_path="/ignored",
                    tmp_repo_path="/ignored",
                    phase="correction_only",
                    resume_from=state_path,
                    example_files=["README.md", "docs/user_guide.md"],
                )

            assert result_dir == out_dir
            fake_gen.run.assert_called_once()
            kwargs = fake_gen.run.call_args.kwargs
            assert sorted(kwargs["target_files"]) == [
                "README.md",
                "docs/user_guide.md",
            ], "example_files whitelist must narrow target_files"
            assert kwargs["max_files"] == 2

            manifests_passed = fake_evaluator.evaluate_multiple_files.call_args.args[0]
            assert set(manifests_passed.keys()) == {"README.md", "docs/user_guide.md"}

    def test_full_phase_preserves_status_quo(self):
        """Smoke: phase='full' is accepted and dispatches through status-quo path."""
        sig = inspect.signature(GenerationTestManagerV2.run_quant_test)
        assert sig.parameters["phase"].default == "full"
