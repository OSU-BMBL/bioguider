"""Pattern B + D regression tests for README project-level classification.

The project-level README is the repository-root one; nested READMEs (data dirs,
test fixtures, vendored sub-packages, docs/) must be folder-level so they are not
scored as the project's documentation (Pattern B), while the root README must
always be project-level so a scorecard renders (Pattern D).
"""
import os

from bioguider.agents.evaluation_readme_task import EvaluationREADMETask


def _classify(tmp_path, files):
    for f in files:
        p = tmp_path / f
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(f"# {f}\nreal readme content\n")
    task = EvaluationREADMETask(
        llm=None, repo_path=str(tmp_path), gitignore_path=str(tmp_path / ".gitignore"),
    )
    res, _ = task._project_level_evaluate(files)
    return {f: res[f]["project_level"] for f in files}


def test_nested_readmes_demoted_pattern_b(tmp_path):
    # GraphST-like: root README is project-level; data-dir & vendored readmes are not
    out = _classify(tmp_path, [
        "README.md", "Data/README.md", "GraphST/readme.txt",
        "tests/test_data/temp/README.md", "alphafold/README.md",
    ])
    assert out["README.md"] is True
    assert out["Data/README.md"] is False
    assert out["GraphST/readme.txt"] is False
    assert out["tests/test_data/temp/README.md"] is False
    assert out["alphafold/README.md"] is False


def test_root_readme_always_project_level_pattern_d(tmp_path):
    # NewWave-script-like: the only README is at root and must be project-level
    out = _classify(tmp_path, ["README.md"])
    assert out["README.md"] is True


def test_fallback_promotes_shallowest_when_no_root(tmp_path):
    # No root README -> shallowest readable candidate is promoted
    out = _classify(tmp_path, ["docs/README.md", "docs/notebooks/README.rst"])
    assert out["docs/README.md"] is True
    assert out["docs/notebooks/README.rst"] is False


def test_unreadable_or_empty_is_not_project_level(tmp_path):
    (tmp_path / "README.md").write_text("")          # empty root readme
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "README.md").write_text("# real\ncontent")
    task = EvaluationREADMETask(
        llm=None, repo_path=str(tmp_path), gitignore_path=str(tmp_path / ".gitignore"),
    )
    res, _ = task._project_level_evaluate(["README.md", "sub/README.md"])
    # empty root is skipped; fallback promotes the readable nested one
    assert res["README.md"]["project_level"] is False
    assert res["sub/README.md"]["project_level"] is True
