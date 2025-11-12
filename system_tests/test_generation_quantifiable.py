import os
import pytest
from bioguider.managers.generation_test_manager_v2 import GenerationTestManagerV2

def test_generation_quantifiable_seurat(llm, step_callback):
    """
    Test error injection and automatic fixing for Seurat documentation.
    
    V2 improvements:
    - Injects errors into multiple file types (README, tutorials, userguides, installation)
    - Increased error counts per level (5, 15, 30)
    - Comprehensive statistics by file type and error category
    - Simplified reporting (fixed vs unchanged)
    - Tracks all injected errors across all files
    """
    report_path = "logs/evaluation_report_github_satijalab_seurat.json"
    baseline_repo_path = "data/.adalflow/repos/satijalab_seurat"
    tmp_repo_path = "outputs/_tmp_satijalab_seurat"

    mgr = GenerationTestManagerV2(llm, step_callback)
    # Low, Mid, High injection levels with increased error counts
    # Start with just low for testing
    suite = mgr.run_quant_suite(
        report_path,
        baseline_repo_path,
        tmp_repo_path,
        levels={"low": 5},  # Test low first, then add mid and high
    )

    for level, out_dir in suite.items():
        assert os.path.isdir(out_dir), f"Output directory not created for {level}"
        assert os.path.isfile(os.path.join(out_dir, "GEN_TEST_RESULTS.json")), f"Test results JSON missing for {level}"
        assert os.path.isfile(os.path.join(out_dir, "GEN_TEST_REPORT.md")), f"Test report MD missing for {level}"
        assert os.path.isfile(os.path.join(out_dir, "README.original.md")), f"Original README missing for {level}"
        assert os.path.isfile(os.path.join(out_dir, "README.corrupted.md")), f"Corrupted README missing for {level}"
        assert os.path.isfile(os.path.join(out_dir, "README.md")), f"Fixed README missing for {level}"

@pytest.mark.skip(reason="deprecated")
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
        levels={"low": 15, "mid": 50, "high": 100},
    )

    for level, out_dir in suite.items():
        assert os.path.isdir(out_dir)
        assert os.path.isfile(os.path.join(out_dir, "GEN_TEST_RESULTS.json"))
        assert os.path.isfile(os.path.join(out_dir, "GEN_TEST_REPORT.md"))
        assert os.path.isfile(os.path.join(out_dir, "README.original.md"))
        assert os.path.isfile(os.path.join(out_dir, "README.corrupted.md"))
        assert os.path.isfile(os.path.join(out_dir, "README.md"))


