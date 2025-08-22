import pytest

from bioguider.agents.evaluation_userguide_task import EvaluationUserGuideTask
from bioguider.managers.evaluation_manager import EvaluationManager

def test_EvaluationInstallationTask_RepoAgent(llm, step_callback, root_path):
    files = ["README_CN.md", "requirements.txt", "display/README_DISPLAY.md"]

    task = EvaluationUserGuideTask(
        llm=llm,
        repo_path=f"{root_path}/RepoAgent",
        gitignore_path=f"{root_path}/RepoAgent/.gitignore",
        step_callback=step_callback,
    )
    # evaluations, token_usage, files = task._evaluate(files)
    files = task._collect_files()

    # assert evaluations is not None
    assert files is not None and len(files) > 0