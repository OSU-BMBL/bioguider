import os

from bioguider.managers.generation_manager import DocumentationGenerationManager


def test_DocumentationGenerationManager_on_seurat_report(llm, step_callback):
    report_path = "logs/seurat_evaluation_results_20250829_112435.json"
    repo_path = "data/.adalflow/repos/satijalab_seurat"

    mgr = DocumentationGenerationManager(llm, step_callback)
    mgr.prepare_repo(repo_path)

    out_dir = mgr.run(report_path=report_path, repo_path=repo_path)
    assert os.path.isdir(out_dir)

    manifest_path = os.path.join(out_dir, "manifest.json")
    assert os.path.isfile(manifest_path)

    # At least ensures pipeline produced outputs directory; documents may be skipped if source files missing
    # If revised README exists, check it is non-empty
    readme_path = os.path.join(out_dir, "README.md")
    if os.path.isfile(readme_path):
        with open(readme_path, "r", encoding="utf-8") as fobj:
            content = fobj.read()
            assert len(content) > 0


