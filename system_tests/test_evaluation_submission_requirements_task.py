
import pytest

from bioguider.agents.evaluation_submission_requirements_task import EvaluationSubmissionRequirementsTask

@pytest.mark.skip()
def test_EvaluationSubmissionRequirementsTask_POPPER(llm, step_callback, root_path):
    task = EvaluationSubmissionRequirementsTask(
        llm=llm,
        repo_path=f"{root_path}/POPPER",
        gitignore_path=f"{root_path}/POPPER/.gitignore",
        step_callback=step_callback,
    )

    files = task._collect_software_package_content()
    assert len(files) == 3

@pytest.mark.skip()
def test_EvaluationSubmissionRequirementsTask_RepoAgent(llm, step_callback, root_path):
    task = EvaluationSubmissionRequirementsTask(
        llm=llm,
        repo_path=f"{root_path}/seurat",
        gitignore_path=f"{root_path}/seurat/.gitignore",
        step_callback=step_callback,
    )

    files = task._collect_software_package_content()
    assert len(files) == 3

# @pytest.mark.skip()
def test_EvaluationSubmissionRequirementsTask_DemoInstructions_RepoAgent(llm, step_callback, root_path):
    task = EvaluationSubmissionRequirementsTask(
        llm=llm,
        repo_path=f"{root_path}/POPPER",
        gitignore_path=f"{root_path}/POPPER/.gitignore",
        step_callback=step_callback,
        readme_files_evaluation={"README.md": {"evaluation": {"project_level": True}}},
        installation_evaluation={},
        installation_files=["README.md", "requirements.txt", "Dockerfile"]
    )

    files = task._evaluatie_demo_instructions()
    assert len(files) == 3


