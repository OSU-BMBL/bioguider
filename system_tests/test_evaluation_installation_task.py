import pytest

from bioguider.agents.evaluation_installation_task import EvaluationInstallationTask
from bioguider.managers.evaluation_manager import EvaluationManager

def test_EvaluationInstallationTask_RepoAgent(llm, step_callback, root_path):
    files = ["README.md", "README_CN.md", "requirements.txt"]

    task = EvaluationInstallationTask(
        llm=llm,
        repo_path=f"{root_path}/RepoAgent",
        gitignore_path=f"{root_path}/RepoAgent/.gitignore",
        step_callback=step_callback,
    )
    evaluations = task.evaluate(files)

    assert evaluations is not None
    assert files is not None and len(files) > 0

