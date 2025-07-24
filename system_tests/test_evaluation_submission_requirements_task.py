
import pytest

from bioguider.agents.evaluation_submission_requirements_task import (
    EvaluationSubmissionRequirementsTask,
)
from bioguider.utils.constants import EvaluationInstallationResult, EvaluationREADMEResult, StructuredEvaluationInstallationResult, StructuredEvaluationREADMEResult

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
        readme_files_evaluation={"README.md": EvaluationREADMEResult(
            project_level=True,
            structured_evaluation=StructuredEvaluationREADMEResult(
                available_score=True,
                readability_score="Good",
                readability_suggestions="",
                project_purpose_score=True,
                project_purpose_suggestions="",
                hardware_and_software_spec_score="Good",
                hardware_and_software_spec_suggestions="",
                dependency_score="Good",
                dependency_suggestions="",
                license_score=False,
                license_suggestions="",
                contributor_author_score=True,
                overall_score="Good",
            ),
            free_evaluation=None,
            structured_reasoning_process=None,
            free_reasoning_process=None,
        )},
        installation_evaluation=EvaluationInstallationResult(
            structured_evaluation=StructuredEvaluationInstallationResult(
                install_available=True,
                install_tutorial=True,
                dependency_number=1,
                dependency_suggestions="",
                compatible_os=True,
                overall_score="Good",
            ),
            free_evaluation=None,
            structured_reasoning_process=None,
            free_reasoning_process=None,
        ),
        installation_files=["README.md", "requirements.txt", "Dockerfile"]
    )

    evaluation, files = task._evaluatie_demo_instructions()
    assert len(files) == 3


