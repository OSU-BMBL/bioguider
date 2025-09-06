import os

from bioguider.managers.generation_test_manager import GenerationTestManager


def test_generation_quantifiable(llm, step_callback):
    report_path = "logs/seurat_evaluation_results_20250829_112435.json"
    baseline_repo_path = "data/.adalflow/repos/satijalab_seurat"
    tmp_repo_path = "outputs/_tmp_satijalab_seurat"

    mgr = GenerationTestManager(llm, step_callback)
    out_dir = mgr.run_quant_test(report_path, baseline_repo_path, tmp_repo_path)

    assert os.path.isdir(out_dir)
    assert os.path.isfile(os.path.join(out_dir, "GEN_TEST_RESULTS.json"))
    assert os.path.isfile(os.path.join(out_dir, "GEN_TEST_REPORT.md"))


