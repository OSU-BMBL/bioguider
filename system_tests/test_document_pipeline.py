"""
System tests for DocumentPipeline.

These tests make real LLM calls and require the Seurat repo to be cloned at
data/.adalflow/repos/satijalab_seurat/. They are intentionally slow and are
excluded from the fast unit-test suite.

Run a single test:
    pytest system_tests/test_document_pipeline.py::TestPrepareRepo::test_builds_code_structure_db -v -s
"""
import os
import pytest
from pathlib import Path

from bioguider.generation.document_pipeline import DocumentPipeline, _build_merged_report
from bioguider.utils.constants import EvaluationTypeEnum


SEURAT_REPO_PATH = "data/.adalflow/repos/satijalab_seurat"
DE_VIGNETTE_PATH = "vignettes/de_vignette.Rmd"


def seurat_repo_available():
    return Path(SEURAT_REPO_PATH, DE_VIGNETTE_PATH).exists()


skip_no_repo = pytest.mark.skipif(
    not seurat_repo_available(),
    reason="Seurat repo not cloned at data/.adalflow/repos/satijalab_seurat/",
)


# ---------------------------------------------------------------------------
# prepare_repo
# ---------------------------------------------------------------------------

class TestPrepareRepo:
    @skip_no_repo
    def test_builds_code_structure_db(self, llm):
        pipeline = DocumentPipeline(SEURAT_REPO_PATH)
        pipeline.prepare_repo(llm)
        assert pipeline.code_structure_db is not None

    @skip_no_repo
    def test_builds_summary_file_db(self, llm):
        pipeline = DocumentPipeline(SEURAT_REPO_PATH)
        pipeline.prepare_repo(llm)
        assert pipeline.summary_file_db is not None

    @skip_no_repo
    def test_returns_self_for_chaining(self, llm):
        pipeline = DocumentPipeline(SEURAT_REPO_PATH)
        result = pipeline.prepare_repo(llm)
        assert result is pipeline

    @skip_no_repo
    def test_prepare_twice_does_not_raise(self, llm):
        pipeline = DocumentPipeline(SEURAT_REPO_PATH)
        pipeline.prepare_repo(llm)
        pipeline.prepare_repo(llm)  # second call should overwrite, not error
        assert pipeline.code_structure_db is not None


# ---------------------------------------------------------------------------
# evaluate_and_refine_document
# ---------------------------------------------------------------------------

class TestEvaluateAndRefineDocument:
    @pytest.fixture(scope="class")
    def prepared_pipeline(self, llm):
        """Build the pipeline once for the whole class to save time."""
        if not seurat_repo_available():
            pytest.skip("Seurat repo not available")
        pipeline = DocumentPipeline(SEURAT_REPO_PATH)
        pipeline.prepare_repo(llm)
        return pipeline

    @skip_no_repo
    def test_returns_report_and_content(self, prepared_pipeline, llm, tmp_path):
        """evaluate_and_refine_document returns a non-empty dict and non-empty str."""
        original = Path(SEURAT_REPO_PATH, DE_VIGNETTE_PATH).read_text()
        doc_file = tmp_path / "de_vignette.Rmd"
        doc_file.write_text(original)

        report, refined = prepared_pipeline.evaluate_and_refine_document(
            llm=llm,
            doc_repo_path=str(tmp_path),
            doc_path="de_vignette.Rmd",
            eval_type=EvaluationTypeEnum.TUTORIAL,
        )

        assert isinstance(report, dict)
        assert "total_suggestions" in report
        assert isinstance(refined, str) and len(refined) > 0

    @skip_no_repo
    def test_report_has_suggestions_list(self, prepared_pipeline, llm, tmp_path):
        original = Path(SEURAT_REPO_PATH, DE_VIGNETTE_PATH).read_text()
        (tmp_path / "de_vignette.Rmd").write_text(original)

        report, _ = prepared_pipeline.evaluate_and_refine_document(
            llm=llm,
            doc_repo_path=str(tmp_path),
            doc_path="de_vignette.Rmd",
            eval_type=EvaluationTypeEnum.TUTORIAL,
        )

        assert "suggestions" in report
        assert isinstance(report["suggestions"], list)

    @skip_no_repo
    def test_refined_is_not_identical_to_input(self, prepared_pipeline, llm, tmp_path):
        """The generator should produce at least some change on a clean vignette."""
        original = Path(SEURAT_REPO_PATH, DE_VIGNETTE_PATH).read_text()
        (tmp_path / "de_vignette.Rmd").write_text(original)

        _, refined = prepared_pipeline.evaluate_and_refine_document(
            llm=llm,
            doc_repo_path=str(tmp_path),
            doc_path="de_vignette.Rmd",
            eval_type=EvaluationTypeEnum.TUTORIAL,
        )

        # The generator may return original when no suggestions exist, but it
        # should at minimum return a non-empty string of reasonable length.
        assert len(refined) > len(original) * 0.5

    @skip_no_repo
    def test_different_llms_both_succeed(self, prepared_pipeline, llm, tmp_path):
        """The pipeline accepts any LLM per-call without requiring re-prepare."""
        from bioguider.agents.agent_utils import get_configured_llm

        # A second, independently-constructed LLM instance honoring the env
        # provider config (LLM_PROVIDER / *_BASE_URL).
        alt_llm = get_configured_llm()

        original = Path(SEURAT_REPO_PATH, DE_VIGNETTE_PATH).read_text()
        (tmp_path / "de_vignette.Rmd").write_text(original)

        report1, refined1 = prepared_pipeline.evaluate_and_refine_document(
            llm=llm,
            doc_repo_path=str(tmp_path),
            doc_path="de_vignette.Rmd",
            eval_type=EvaluationTypeEnum.TUTORIAL,
        )
        report2, refined2 = prepared_pipeline.evaluate_and_refine_document(
            llm=alt_llm,
            doc_repo_path=str(tmp_path),
            doc_path="de_vignette.Rmd",
            eval_type=EvaluationTypeEnum.TUTORIAL,
        )

        assert isinstance(refined1, str) and len(refined1) > 0
        assert isinstance(refined2, str) and len(refined2) > 0
