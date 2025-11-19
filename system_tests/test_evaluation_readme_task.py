import pytest

from bioguider.agents.evaluation_readme_task import EvaluationREADMETask

@pytest.mark.skip()
def test_EvaluationReadmeTask_RepoAgent(llm, step_callback, root_path):
    files = ["README.md"]

    task = EvaluationREADMETask(
        llm=llm,
        repo_path=f"{root_path}/RepoAgent",
        gitignore_path=f"{root_path}/RepoAgent/.gitignore",
        step_callback=step_callback,
    )
    evaluations, token_usage, files = task._evaluate(files)

    assert evaluations is not None
    assert files is not None and len(files) > 0

    for file, evaluation in evaluations.items():
        assert evaluation is not None
        assert evaluation.structured_evaluation.overall_score is not None
        assert evaluation.structured_evaluation.overall_score > 60 and evaluation.structured_evaluation.overall_score <= 100



def test_EvaluationReadmeTask_deepmaps(llm, step_callback, root_path):
    task = EvaluationREADMETask(
        llm=llm,
        repo_path=f"{root_path}/deepmaps",
        gitignore_path=f"{root_path}/deepmaps/.gitignore",
        step_callback=step_callback,
    )
    evaluations, files = task.evaluate()
    assert evaluations is not None
    assert files is not None and len(files) > 0
    for file, evaluation in evaluations.items():
        assert evaluation is not None