
import pytest

from bioguider.agents.evaluation_task import EvaluationREADMETask, EvaluationTutorialTask

@pytest.mark.skip()
def test_EvaluationREADMETask(llm, step_callback):
    task = EvaluationREADMETask(
        llm,
        repo_path="./data/repos/POPPER",
        gitignore_path="./data/repos/POPPER/.gitignore",
        step_callback=step_callback,
    )
    res = task.evaluate(
        files=["README.md"],
    )
    assert res is not None

@pytest.mark.skip()
def test_EvaluationREADMETask_RepoAgent(llm, step_callback):
    task = EvaluationREADMETask(
        llm,
        repo_path="./data/repos/RepoAgent",
        gitignore_path="./data/repos/RepoAgent/.gitignore",
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
    res = task.evaluate(
        files=["README.md"],
    )
    assert res is not None

@pytest.mark.skip()
def test_EvaluationTutorialTask_POPPER(llm, step_callback):
    task = EvaluationTutorialTask(
        llm,
        repo_path="./data/repos/POPPER",
        gitignore_path="./data/repos/POPPER/.gitignore",
        step_callback=step_callback,
    )
    res = task.evaluate(
        # files=["README.md", "setup.py", "Dockerfile"],
        files=["README.md"],
    )
    assert res is not None
