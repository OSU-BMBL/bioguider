
import pytest

from bioguider.agents.evaluation_task import EvaluationTutorialTask
from bioguider.agents.evaluation_readme_task import EvaluationREADMETask

# @pytest.mark.skip()
def test_EvaluationREADMETask(llm, step_callback, root_path):
    task = EvaluationREADMETask(
        llm,
        repo_path=f"{root_path}/POPPER",
        gitignore_path=f"{root_path}/POPPER/.gitignore",
        step_callback=step_callback,
    )
    res, files = task.evaluate()
    assert res is not None
    assert len(files) > 0

@pytest.mark.skip()
def test_EvaluationREADMETask_RepoAgent(llm, step_callback, root_path):
    task = EvaluationREADMETask(
        llm,
        repo_path=f"{root_path}/RepoAgent",
        gitignore_path=f"{root_path}/RepoAgent/.gitignore",
        step_callback=step_callback,
    )
    res = task.evaluate()
    assert res is not None

@pytest.mark.skip()
def test_EvaluationREADMETask_BMBL_analysis_notebooks(llm, step_callback):
    task = EvaluationREADMETask(
        llm,
        repo_path="./data/repos/BMBL-analysis-notebooks",
        gitignore_path="./data/repos/BMBL-analysis-notebooks/.gitignore",
        step_callback=step_callback,
    )
    res = task.evaluate()
    assert res is not None

@pytest.mark.skip()
def test_EvaluationTutorialTask_POPPER(llm, step_callback):
    task = EvaluationTutorialTask(
        llm,
        repo_path="./data/repos/POPPER",
        gitignore_path="./data/repos/POPPER/.gitignore",
        step_callback=step_callback,
    )
    res = task.evaluate()
    assert res is not None
