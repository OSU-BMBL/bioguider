"""
Unit tests for bioguider/generation/document_pipeline.py.

All tests run without LLM calls or disk I/O by using mock objects.
"""
import json
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from bioguider.utils.constants import EvaluationTypeEnum
from bioguider.generation.document_pipeline import (
    DocumentPipeline,
    _build_merged_report,
    _run_evaluation,
    _serialise_eval_results,
)


# ---------------------------------------------------------------------------
# Helpers — build lightweight mock eval results
# ---------------------------------------------------------------------------

def _make_tutorial_result(errors=None, readability=None, setup=None,
                           reproducibility=None, structure=None,
                           code_quality=None, result_verification=None,
                           performance=None):
    te = MagicMock()
    te.readability_errors_found = errors or []
    te.readability_suggestions = readability or []
    te.setup_and_dependencies_suggestions = setup or []
    te.reproducibility_suggestions = reproducibility or []
    te.structure_and_navigation_suggestions = structure or []
    te.executable_code_quality_suggestions = code_quality or []
    te.result_verification_suggestions = result_verification or []
    te.performance_and_resource_notes_suggestions = performance or []
    result = MagicMock()
    result.tutorial_evaluation = te
    return result


def _make_userguide_result(errors=None, readability=None,
                            context=None, error_handling=None,
                            consistency_development=None,
                            consistency_evaluation=...):
    ug = MagicMock()
    ug.readability_errors_found = errors or []
    ug.readability_suggestions = readability or []
    ug.context_and_purpose_suggestions = context or []
    ug.error_handling_suggestions = error_handling or []
    result = MagicMock()
    result.user_guide_evaluation = ug
    if consistency_evaluation is ...:
        ce = MagicMock()
        ce.development = consistency_development or []
        result.consistency_evaluation = ce
    else:
        result.consistency_evaluation = consistency_evaluation
    return result


def _attach_consistency(result, development=None, *, none=False):
    """Attach a ConsistencyEvaluationResult-shaped mock to *result*."""
    if none:
        result.consistency_evaluation = None
        return result
    ce = MagicMock()
    ce.development = development or []
    result.consistency_evaluation = ce
    return result


# ---------------------------------------------------------------------------
# _build_merged_report — TUTORIAL
# ---------------------------------------------------------------------------

class TestBuildMergedReportTutorial:
    def test_all_category_fields_extracted(self):
        result = _make_tutorial_result(
            errors=["typo: analysi → analysis"],
            readability=["Simplify sentence X"],
            setup=["Add dependency Y"],
            reproducibility=["Specify seed"],
            structure=["Add section header"],
            code_quality=["Fix hardcoded path"],
            result_verification=["Show expected output"],
            performance=["Note memory requirement"],
        )
        report = _build_merged_report(
            {"doc.Rmd": result}, "doc.Rmd", EvaluationTypeEnum.TUTORIAL
        )
        assert report["total_suggestions"] == 8
        categories = [s["category"] for s in report["suggestions"]]
        assert "readability_errors" in categories
        assert "readability" in categories
        assert "setup" in categories
        assert "reproducibility" in categories
        assert "structure" in categories
        assert "code_quality" in categories
        assert "result_verification" in categories
        assert "performance" in categories

    def test_suggestion_numbers_are_sequential(self):
        result = _make_tutorial_result(
            errors=["e1", "e2"], readability=["r1"]
        )
        report = _build_merged_report(
            {"doc.Rmd": result}, "doc.Rmd", EvaluationTypeEnum.TUTORIAL
        )
        numbers = [s["suggestion_number"] for s in report["suggestions"]]
        assert numbers == list(range(1, len(numbers) + 1))

    def test_empty_when_doc_path_not_in_results(self):
        result = _make_tutorial_result(errors=["e1"])
        report = _build_merged_report(
            {"other.Rmd": result}, "doc.Rmd", EvaluationTypeEnum.TUTORIAL
        )
        assert report["total_suggestions"] == 0
        assert report["suggestions"] == []

    def test_empty_when_eval_results_none(self):
        report = _build_merged_report(None, "doc.Rmd", EvaluationTypeEnum.TUTORIAL)
        assert report["total_suggestions"] == 0

    def test_empty_when_tutorial_evaluation_is_none(self):
        result = MagicMock()
        result.tutorial_evaluation = None
        report = _build_merged_report(
            {"doc.Rmd": result}, "doc.Rmd", EvaluationTypeEnum.TUTORIAL
        )
        assert report["total_suggestions"] == 0

    def test_report_has_integration_instruction(self):
        result = _make_tutorial_result(errors=["e1"])
        report = _build_merged_report(
            {"doc.Rmd": result}, "doc.Rmd", EvaluationTypeEnum.TUTORIAL
        )
        assert "integration_instruction" in report
        assert "1" in report["integration_instruction"]

    def test_suggestion_has_required_keys(self):
        result = _make_tutorial_result(errors=["e1"])
        report = _build_merged_report(
            {"doc.Rmd": result}, "doc.Rmd", EvaluationTypeEnum.TUTORIAL
        )
        s = report["suggestions"][0]
        assert "suggestion_number" in s
        assert "category" in s
        assert "content_guidance" in s


# ---------------------------------------------------------------------------
# _build_merged_report — USERGUIDE
# ---------------------------------------------------------------------------

class TestBuildMergedReportUserguide:
    def test_userguide_fields_extracted(self):
        result = _make_userguide_result(
            errors=["typo: genomis → genomics"],
            readability=["Simplify intro"],
            context=["Add purpose statement"],
            error_handling=["Document exceptions"],
        )
        report = _build_merged_report(
            {"guide.Rmd": result}, "guide.Rmd", EvaluationTypeEnum.USERGUIDE
        )
        assert report["total_suggestions"] == 4
        categories = [s["category"] for s in report["suggestions"]]
        assert "readability_errors" in categories
        assert "readability" in categories
        assert "context_and_purpose" in categories
        assert "error_handling" in categories


# ---------------------------------------------------------------------------
# _build_merged_report — README
# ---------------------------------------------------------------------------

class TestBuildMergedReportReadme:
    def test_readme_structured_suggestions_extracted(self):
        se = MagicMock()
        se.readability_suggestions = "Improve clarity"
        se.project_purpose_suggestions = None
        se.hardware_and_software_spec_suggestions = None
        se.dependency_suggestions = "List all deps"
        se.license_suggestions = None

        fe = MagicMock()
        fe.readability = ["Add examples"]
        fe.project_purpose = None
        fe.hardware_and_software_spec = None
        fe.dependency = None
        fe.license = None
        fe.contributor_author = None
        fe.overall_score = None

        result = MagicMock()
        result.structured_evaluation = se
        result.free_evaluation = fe

        report = _build_merged_report(
            {"README.md": result}, "README.md", EvaluationTypeEnum.README
        )
        assert report["total_suggestions"] == 3  # readability_s + dependency_s + readability_fe

    def test_readme_empty_eval_results(self):
        report = _build_merged_report({}, "README.md", EvaluationTypeEnum.README)
        assert report["total_suggestions"] == 0


# ---------------------------------------------------------------------------
# _build_merged_report — generic fallback
# ---------------------------------------------------------------------------

class TestBuildMergedReportFallback:
    def test_installation_falls_back_to_generic(self):
        report = _build_merged_report(
            {"install.md": {"key": "value"}},
            "install.md",
            EvaluationTypeEnum.INSTALLATION,
        )
        assert report["total_suggestions"] == 1
        assert report["suggestions"][0]["category"] == "general"

    def test_fallback_content_is_json_serializable(self):
        report = _build_merged_report(
            {"f": {"nested": [1, 2]}},
            "f",
            EvaluationTypeEnum.INSTALLATION,
        )
        guidance = report["suggestions"][0]["content_guidance"]
        parsed = json.loads(guidance)
        assert "f" in parsed


# ---------------------------------------------------------------------------
# DocumentPipeline — construction
# ---------------------------------------------------------------------------

class TestDocumentPipelineConstruction:
    def test_stores_repo_path(self):
        p = DocumentPipeline("/some/repo")
        assert p.repo_path == "/some/repo"

    def test_dbs_none_before_prepare(self):
        p = DocumentPipeline("/some/repo")
        assert p.code_structure_db is None
        assert p.summary_file_db is None

    def test_prepare_returns_self_for_chaining(self):
        p = DocumentPipeline("/some/repo")
        mock_llm = MagicMock()
        mock_manager = MagicMock()
        mock_manager.code_structure_db = object()
        mock_manager.summary_file_db = object()
        with patch(
            "bioguider.generation.document_pipeline.DocumentPipeline.prepare_repo",
            return_value=p,
        ):
            result = p.prepare_repo(mock_llm)
        assert result is p

    def test_prepare_repo_populates_dbs(self):
        p = DocumentPipeline("/some/repo")
        mock_llm = MagicMock()
        fake_code_db = object()
        fake_summary_db = object()
        mock_manager = MagicMock()
        mock_manager.code_structure_db = fake_code_db
        mock_manager.summary_file_db = fake_summary_db

        # EvaluationManager is imported lazily inside prepare_repo, so patch
        # it at the source module rather than on document_pipeline.
        with patch(
            "bioguider.managers.evaluation_manager.EvaluationManager",
            return_value=mock_manager,
        ):
            p.prepare_repo(mock_llm)

        assert p.code_structure_db is fake_code_db
        assert p.summary_file_db is fake_summary_db


# ---------------------------------------------------------------------------
# _run_evaluation — dispatch
# ---------------------------------------------------------------------------

class TestRunEvaluationDispatch:
    def _make_meta(self):
        from bioguider.utils.constants import ProjectMetadata, ProjectTypeEnum, PrimaryLanguageEnum
        return ProjectMetadata(
            url="/repo",
            project_type=ProjectTypeEnum.unknown,
            primary_language=PrimaryLanguageEnum.unknown,
        )

    def _call(self, eval_type, task_class_path, task_class_name):
        mock_llm = MagicMock()
        meta = self._make_meta()
        fake_results = ({"doc.Rmd": MagicMock()}, ["doc.Rmd"])
        mock_task = MagicMock()
        mock_task.evaluate.return_value = fake_results

        with patch(task_class_path) as MockTask:
            MockTask.return_value = mock_task
            result = _run_evaluation(
                llm=mock_llm,
                repo_path="/repo",
                gitignore_path="/repo/.gitignore",
                doc_path="doc.Rmd",
                meta=meta,
                eval_type=eval_type,
            )
        MockTask.assert_called_once()
        assert result == fake_results

    def test_dispatches_tutorial(self):
        self._call(
            EvaluationTypeEnum.TUTORIAL,
            "bioguider.agents.evaluation_tutorial_task.EvaluationTutorialTask",
            "EvaluationTutorialTask",
        )

    def test_dispatches_readme(self):
        self._call(
            EvaluationTypeEnum.README,
            "bioguider.agents.evaluation_readme_task.EvaluationREADMETask",
            "EvaluationREADMETask",
        )

    def test_dispatches_installation(self):
        self._call(
            EvaluationTypeEnum.INSTALLATION,
            "bioguider.agents.evaluation_installation_task.EvaluationInstallationTask",
            "EvaluationInstallationTask",
        )

    def test_dispatches_userguide(self):
        self._call(
            EvaluationTypeEnum.USERGUIDE,
            "bioguider.agents.evaluation_userguide_task.EvaluationUserGuideTask",
            "EvaluationUserGuideTask",
        )

    def test_raises_for_unsupported_type(self):
        mock_llm = MagicMock()
        meta = self._make_meta()
        with pytest.raises(ValueError, match="Unsupported eval_type"):
            _run_evaluation(
                llm=mock_llm,
                repo_path="/repo",
                gitignore_path="/repo/.gitignore",
                doc_path="doc.Rmd",
                meta=meta,
                eval_type=EvaluationTypeEnum.SUBMISSION_REQUIREMENTS,
            )

    def test_passes_code_structure_db_to_tutorial(self):
        mock_llm = MagicMock()
        meta = self._make_meta()
        fake_db = object()
        mock_task = MagicMock()
        mock_task.evaluate.return_value = ({}, [])

        with patch(
            "bioguider.agents.evaluation_tutorial_task.EvaluationTutorialTask"
        ) as MockTask:
            MockTask.return_value = mock_task
            _run_evaluation(
                llm=mock_llm,
                repo_path="/repo",
                gitignore_path="/repo/.gitignore",
                doc_path="doc.Rmd",
                meta=meta,
                eval_type=EvaluationTypeEnum.TUTORIAL,
                code_structure_db=fake_db,
            )
        _, kwargs = MockTask.call_args
        assert kwargs.get("code_structure_db") is fake_db

    def test_passes_summary_file_db_to_tutorial(self):
        mock_llm = MagicMock()
        meta = self._make_meta()
        fake_db = object()
        mock_task = MagicMock()
        mock_task.evaluate.return_value = ({}, [])

        with patch(
            "bioguider.agents.evaluation_tutorial_task.EvaluationTutorialTask"
        ) as MockTask:
            MockTask.return_value = mock_task
            _run_evaluation(
                llm=mock_llm,
                repo_path="/repo",
                gitignore_path="/repo/.gitignore",
                doc_path="doc.Rmd",
                meta=meta,
                eval_type=EvaluationTypeEnum.TUTORIAL,
                summary_file_db=fake_db,
            )
        _, kwargs = MockTask.call_args
        assert kwargs.get("summarized_files_db") is fake_db


# ---------------------------------------------------------------------------
# DocumentPipeline.evaluate_and_refine_document — integration (mocked)
# ---------------------------------------------------------------------------

class TestEvaluateAndRefineDocumentMocked:
    def test_evaluate_and_refine_calls_generator(self, tmp_path):
        """evaluate_and_refine_document calls LLMContentGenerator with the right args."""
        doc_content = "---\ntitle: Test\n---\n\n# Intro\n\nSome text.\n"
        doc_file = tmp_path / "test.Rmd"
        doc_file.write_text(doc_content)

        pipeline = DocumentPipeline(str(tmp_path))
        mock_llm = MagicMock()
        fake_result = _make_tutorial_result(errors=["typo: foo → bar"])
        fake_eval_results = {"test.Rmd": fake_result}

        with patch(
            "bioguider.generation.document_pipeline._run_evaluation",
            return_value=(fake_eval_results, ["test.Rmd"]),
        ), patch(
            "bioguider.generation.document_pipeline.LLMContentGenerator"
        ) as MockGen:
            mock_gen_instance = MagicMock()
            mock_gen_instance.generate_full_document.return_value = ("refined content", {})
            MockGen.return_value = mock_gen_instance

            report, refined = pipeline.evaluate_and_refine_document(
                llm=mock_llm,
                doc_repo_path=str(tmp_path),
                doc_path="test.Rmd",
                eval_type=EvaluationTypeEnum.TUTORIAL,
            )

        MockGen.assert_called_once_with(mock_llm)
        mock_gen_instance.generate_full_document.assert_called_once()
        assert refined == "refined content"
        assert report["total_suggestions"] == 1

    def test_evaluate_and_refine_falls_back_when_generator_returns_empty(self, tmp_path):
        """Falls back to original_content when generator returns empty string."""
        doc_content = "original content"
        (tmp_path / "doc.Rmd").write_text(doc_content)

        pipeline = DocumentPipeline(str(tmp_path))
        mock_llm = MagicMock()

        with patch(
            "bioguider.generation.document_pipeline._run_evaluation",
            return_value=({}, []),
        ), patch(
            "bioguider.generation.document_pipeline.LLMContentGenerator"
        ) as MockGen:
            mock_gen_instance = MagicMock()
            mock_gen_instance.generate_full_document.return_value = ("", {})
            MockGen.return_value = mock_gen_instance

            _, refined = pipeline.evaluate_and_refine_document(
                llm=mock_llm,
                doc_repo_path=str(tmp_path),
                doc_path="doc.Rmd",
                eval_type=EvaluationTypeEnum.TUTORIAL,
            )

        assert refined == doc_content


# ---------------------------------------------------------------------------
# _serialise_eval_results — direct
# ---------------------------------------------------------------------------


class _PydanticLike:
    """Stub mirroring the pydantic-v2 ``model_dump`` surface used by the helper."""
    def __init__(self, payload):
        self._payload = payload

    def model_dump(self):
        return dict(self._payload)


class _PydanticRaises:
    """Stub whose ``model_dump`` raises — exercises the fallback branch."""
    def model_dump(self):
        raise RuntimeError("boom")


class TestBuildMergedReportConsistency:
    """``consistency_evaluation.development`` must flow into the merged
    report as ``"consistency"`` suggestions — otherwise the consistency
    task's findings (CLI/code inconsistencies, mismatched names) never
    reach the generator."""

    def test_userguide_consistency_development_emitted(self):
        result = _make_userguide_result(
            readability=["clarify intro"],
            consistency_development=[
                "Documentation passes --cores to pharokka_multiplotter.py, "
                "but the parser does not define this option.",
                "Function name `Plotr` referenced in prose; actual name is `Plotter`.",
            ],
        )
        report = _build_merged_report(
            {"doc.md": result}, "doc.md", EvaluationTypeEnum.USERGUIDE
        )
        consistency_items = [s for s in report["suggestions"] if s["category"] == "consistency"]
        assert len(consistency_items) == 2
        joined = " ".join(s["content_guidance"] for s in consistency_items)
        assert "--cores" in joined
        assert "Plotr" in joined
        # Existing categories still flow through.
        assert any(s["category"] == "readability" for s in report["suggestions"])

    def test_tutorial_consistency_development_emitted(self):
        result = _make_tutorial_result(readability=["clarify intro"])
        _attach_consistency(result, development=["docstring mismatch: foo()"])
        report = _build_merged_report(
            {"doc.Rmd": result}, "doc.Rmd", EvaluationTypeEnum.TUTORIAL
        )
        consistency_items = [s for s in report["suggestions"] if s["category"] == "consistency"]
        assert len(consistency_items) == 1
        assert "foo()" in consistency_items[0]["content_guidance"]

    def test_consistency_evaluation_none_is_ignored(self):
        result = _make_userguide_result(
            readability=["clarify intro"],
            consistency_evaluation=None,
        )
        report = _build_merged_report(
            {"doc.md": result}, "doc.md", EvaluationTypeEnum.USERGUIDE
        )
        assert not any(s["category"] == "consistency" for s in report["suggestions"])
        # Non-consistency suggestions still come through.
        assert any(s["category"] == "readability" for s in report["suggestions"])

    def test_empty_development_emits_zero_consistency_suggestions(self):
        result = _make_userguide_result(
            readability=["clarify intro"],
            consistency_development=[],
        )
        report = _build_merged_report(
            {"doc.md": result}, "doc.md", EvaluationTypeEnum.USERGUIDE
        )
        assert not any(s["category"] == "consistency" for s in report["suggestions"])

    def test_suggestion_categories_filter_keeps_only_consistency(self):
        """When the caller restricts to ``["consistency"]`` only the
        consistency items survive — readability/error_handling/etc. drop."""
        result = _make_userguide_result(
            readability=["clarify intro"],
            error_handling=["add try/except"],
            consistency_development=["undefined --cores flag"],
        )
        report = _build_merged_report(
            {"doc.md": result},
            "doc.md",
            EvaluationTypeEnum.USERGUIDE,
            suggestion_categories=["consistency"],
        )
        cats = {s["category"] for s in report["suggestions"]}
        assert cats == {"consistency"}
        assert report["suggestions"][0]["content_guidance"] == "undefined --cores flag"

    def test_consistency_suggestion_numbers_continue_sequence(self):
        """Suggestion numbers must remain a contiguous sequence across
        category-source boundaries (1, 2, 3, … not 1, 2, 1)."""
        result = _make_userguide_result(
            readability=["a", "b"],
            consistency_development=["c", "d"],
        )
        report = _build_merged_report(
            {"doc.md": result}, "doc.md", EvaluationTypeEnum.USERGUIDE
        )
        nums = [s["suggestion_number"] for s in report["suggestions"]]
        assert nums == list(range(1, len(nums) + 1))

    def test_falsy_development_items_skipped(self):
        """``None``/``""`` entries in development should not become suggestions."""
        result = _make_userguide_result(
            consistency_development=["real finding", "", None, "another finding"],
        )
        report = _build_merged_report(
            {"doc.md": result}, "doc.md", EvaluationTypeEnum.USERGUIDE
        )
        consistency_items = [s for s in report["suggestions"] if s["category"] == "consistency"]
        assert len(consistency_items) == 2


class TestSerialiseEvalResults:
    def test_none_returns_empty_dict(self):
        assert _serialise_eval_results(None) == {}

    def test_empty_returns_empty_dict(self):
        assert _serialise_eval_results({}) == {}

    def test_pydantic_value_is_dumped(self):
        v = _PydanticLike({"score": 87, "assessment": "ok"})
        out = _serialise_eval_results({"foo.md": v})
        assert out == {"foo.md": {"score": 87, "assessment": "ok"}}

    def test_plain_value_passed_through(self):
        out = _serialise_eval_results({"foo.md": {"already": "dict"}})
        assert out == {"foo.md": {"already": "dict"}}

    def test_falls_back_when_model_dump_raises(self):
        v = _PydanticRaises()
        out = _serialise_eval_results({"foo.md": v})
        # Helper must not bubble the exception — it returns the object,
        # leaving the eventual json.dump's default=str to coerce it.
        assert out["foo.md"] is v


# ---------------------------------------------------------------------------
# evaluate_and_refine_document — eval_report_output_path
# ---------------------------------------------------------------------------


def _make_tutorial_result_with_dump(dump_payload, **kwargs):
    """``_make_tutorial_result`` but with ``model_dump`` returning *dump_payload*.

    The MagicMock-based stub supports both:
      - attribute access used by ``_build_merged_report``
        (``result.tutorial_evaluation.readability_errors_found``);
      - ``result.model_dump()`` used by ``_serialise_eval_results``.
    """
    result = _make_tutorial_result(**kwargs)
    result.model_dump.return_value = dump_payload
    return result


class TestEvaluateAndRefineEvalReportOutput:
    def _run(self, tmp_path, *, eval_report_output_path, report_output_path=None):
        doc_content = "# doc\n"
        (tmp_path / "doc.Rmd").write_text(doc_content)

        stub = _make_tutorial_result_with_dump(
            dump_payload={
                "tutorial_evaluation": {
                    "readability_errors_found": ["typo: foo -> bar"],
                    "readability_suggestions": [],
                },
            },
            errors=["typo: foo -> bar"],
        )
        eval_results = {"doc.Rmd": stub}

        pipeline = DocumentPipeline(str(tmp_path))
        with patch(
            "bioguider.generation.document_pipeline._run_evaluation",
            return_value=(eval_results, ["doc.Rmd"]),
        ), patch(
            "bioguider.generation.document_pipeline.LLMContentGenerator"
        ) as MockGen:
            mock_gen_instance = MagicMock()
            mock_gen_instance.generate_full_document.return_value = ("refined", {})
            MockGen.return_value = mock_gen_instance

            return pipeline.evaluate_and_refine_document(
                llm=MagicMock(),
                doc_repo_path=str(tmp_path),
                doc_path="doc.Rmd",
                eval_type=EvaluationTypeEnum.TUTORIAL,
                report_output_path=report_output_path,
                eval_report_output_path=eval_report_output_path,
            )

    def test_writes_json_when_path_given(self, tmp_path):
        eval_path = tmp_path / "out" / "doc.pipeline_eval.json"
        self._run(tmp_path, eval_report_output_path=str(eval_path))
        assert eval_path.exists(), "eval_report_output_path file was not written"
        data = json.loads(eval_path.read_text(encoding="utf-8"))
        assert list(data.keys()) == ["doc.Rmd"]
        assert data["doc.Rmd"]["tutorial_evaluation"]["readability_errors_found"] == [
            "typo: foo -> bar"
        ]

    def test_does_not_write_when_path_none(self, tmp_path):
        sentinel = tmp_path / "no_such_file.json"
        self._run(tmp_path, eval_report_output_path=None)
        assert not sentinel.exists()

    def test_creates_parent_directory(self, tmp_path):
        nested = tmp_path / "a" / "b" / "c" / "eval.json"
        self._run(tmp_path, eval_report_output_path=str(nested))
        assert nested.exists()

    def test_report_and_eval_paths_are_independent(self, tmp_path):
        """Two distinct outputs: the merged generator-input report and the raw
        evaluation. The fields differ in shape and neither overwrites the other."""
        report_path = tmp_path / "merged.json"
        eval_path = tmp_path / "eval.json"
        self._run(
            tmp_path,
            eval_report_output_path=str(eval_path),
            report_output_path=str(report_path),
        )
        merged = json.loads(report_path.read_text(encoding="utf-8"))
        raw = json.loads(eval_path.read_text(encoding="utf-8"))
        # Merged report has the generator-facing shape.
        assert {"total_suggestions", "integration_instruction", "suggestions"} <= set(merged.keys())
        # Raw eval is keyed by doc path.
        assert list(raw.keys()) == ["doc.Rmd"]
        # Different shapes.
        assert "tutorial_evaluation" not in merged
        assert "suggestions" not in raw["doc.Rmd"]
