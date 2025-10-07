import os
import pytest
from bioguider.managers.generation_test_manager import GenerationTestManager

@pytest.mark.skip(reason="Switching to scanpy test")
def test_generation_quantifiable_seurat(llm, step_callback):
    report_path = "logs/seurat_evaluation_results_20250829_112435.json"
    baseline_repo_path = "data/.adalflow/repos/satijalab_seurat"
    tmp_repo_path = "outputs/_tmp_satijalab_seurat"

    mgr = GenerationTestManager(llm, step_callback)
    # Low, Mid, High injection levels
    suite = mgr.run_quant_suite(
        report_path,
        baseline_repo_path,
        tmp_repo_path,
        levels={"low": 2, "mid": 10, "high": 50},
    )

    for level, out_dir in suite.items():
        assert os.path.isdir(out_dir)
        assert os.path.isfile(os.path.join(out_dir, "GEN_TEST_RESULTS.json"))
        assert os.path.isfile(os.path.join(out_dir, "GEN_TEST_REPORT.md"))
        assert os.path.isfile(os.path.join(out_dir, "README.original.md"))
        assert os.path.isfile(os.path.join(out_dir, "README.corrupted.md"))
        assert os.path.isfile(os.path.join(out_dir, "README.md"))


def test_generation_quantifiable_scanpy(llm, step_callback):
    report_path = "logs/scanpy_evaluation_results_20250926.json"
    baseline_repo_path = "data/.adalflow/repos/scverse_scanpy"
    tmp_repo_path = "outputs/_tmp_scverse_scanpy"

    mgr = GenerationTestManager(llm, step_callback)
    # Low, Mid, High injection levels
    suite = mgr.run_quant_suite(
        report_path,
        baseline_repo_path,
        tmp_repo_path,
        levels={"low": 2, "mid": 10, "high": 50},
    )

    for level, out_dir in suite.items():
        assert os.path.isdir(out_dir)
        assert os.path.isfile(os.path.join(out_dir, "GEN_TEST_RESULTS.json"))
        assert os.path.isfile(os.path.join(out_dir, "GEN_TEST_REPORT.md"))
        assert os.path.isfile(os.path.join(out_dir, "README.original.md"))
        assert os.path.isfile(os.path.join(out_dir, "README.corrupted.md"))
        assert os.path.isfile(os.path.join(out_dir, "README.md"))


