
import pytest

from bioguider.agents.evaluation_installation_task import EvaluationInstallationResult, StructuredEvaluationInstallationResult
from bioguider.managers.evaluation_manager import EvaluationManager
from bioguider.utils.constants import PrimaryLanguageEnum, ProjectTypeEnum
from bioguider.agents.evaluation_readme_task import EvaluationREADMEResult, StructuredEvaluationREADMEResult

@pytest.mark.skip()
def test_EvaluationManager(llm, step_callback):
    mgr = EvaluationManager(llm, step_callback)
    mgr.prepare_repo("https://github.com/OSU-BMBL/deepmaps")
    metadata = mgr.identify_project()
    assert str(metadata.project_type.value) == "package"
    assert str(metadata.primary_language.value) == "python"
    assert metadata.repo_name is not None

    mgr.evaluate_readme()

@pytest.mark.skip()
def test_EvaluationManager_on_tutorial(llm, step_callback):
    mgr = EvaluationManager(llm, step_callback)
    mgr.prepare_repo("https://github.com/snap-stanford/POPPER")
    proj_metadata = mgr.identify_project()
    assert proj_metadata.project_type == ProjectTypeEnum.package
    assert proj_metadata.primary_language == PrimaryLanguageEnum.python
    assert proj_metadata.repo_name is not None

    mgr.evaluate_tutorial()


@pytest.mark.skip()
def test_EvaluationManager_on_readme(llm, step_callback):
    json_obj = EvaluationREADMEResult.model_json_schema()
    mgr = EvaluationManager(llm, step_callback)
    mgr.prepare_repo("https://github.com/OSU-BMBL/scGNN2.0")

    evaluations, readme_file = mgr.evaluate_readme()
    assert len(readme_file) > 0
    assert len(evaluations) > 0
    # assert evaluations[0]

@pytest.mark.skip()
def test_EvaluationManager_on_installation(llm, step_callback):
    json_obj = EvaluationREADMEResult.model_json_schema()
    mgr = EvaluationManager(llm, step_callback)
    mgr.prepare_repo("https://github.com/OSU-BMBL/deepmaps")

    evaluation, files = mgr.evaluate_installation()
    assert len(files) > 0

@pytest.mark.skip()
def test_EvaluationManager_on_seurat_requirements(llm, step_callback):
    json_obj = EvaluationREADMEResult.model_json_schema()
    mgr = EvaluationManager(llm, step_callback)
    mgr.prepare_repo("https://github.com/OpenBMB/RepoAgent")

    installation_evaluation, installation_files = mgr.evaluate_installation()
    readme_evaluation, readme_files = mgr.evaluate_readme()
    evaluation, files =  mgr.evaluate_submission_requirements(
        readme_files_evaluation=readme_evaluation,
        installation_files=installation_files,
        installation_evaluation=installation_evaluation,
    )
    assert len(files) > 0

# @pytest.mark.skip()
def test_EvaluationManager_on_seurat_requirements(llm, step_callback):
    import json
    import os
    from datetime import datetime
    
    json_obj = EvaluationREADMEResult.model_json_schema()
    mgr = EvaluationManager(llm, step_callback)
    repo_url = "https://github.com/satijalab/seurat"
    mgr.prepare_repo(repo_url)

    installation_evaluation, installation_files = mgr.evaluate_installation()
    readme_evaluation, readme_files = mgr.evaluate_readme()
    evaluation, files =  mgr.evaluate_submission_requirements(
        readme_files_evaluation=readme_evaluation,
        installation_files=installation_files,
        installation_evaluation=installation_evaluation,
    )
    assert len(files) > 0
    
    # Save evaluation results to a file
    results = {
        "timestamp": datetime.now().isoformat(),
        "repo_url": repo_url,
        "installation_evaluation": installation_evaluation,
        "installation_files": installation_files,
        "readme_evaluation": readme_evaluation,
        "readme_files": readme_files,
        "submission_requirements_evaluation": evaluation,
        "submission_requirements_files": files
    }
    
    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)
    
    # Save to JSON file with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"logs/seurat_evaluation_results_{timestamp}.json"
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"Evaluation results saved to: {filename}")

@pytest.mark.skip()
def test_EvaluationManager_on_POPPER(llm, step_callback):
    json_obj = EvaluationREADMEResult.model_json_schema()
    mgr = EvaluationManager(llm, step_callback)
    mgr.prepare_repo("https://github.com/snap-stanford/POPPER")

    install_evaluation = { "evaluation": EvaluationInstallationResult(
        ease_of_access='The installation instructions are easily accessible within the `README.md` and well-labeled.', 
        score='Good',
        clarity_of_dependency='Clear dependency details are provided in `requirements.txt`, and explicit installation methods using `pip` and `conda` are included.', 
        hardware_requirements="No hardware specifications are mentioned, which is a significant gap given the software's reliance on large models like OpenAI's `Claude`. This omission might hinder users with limited resources.", 
        installation_guide='The step-by-step guide is comprehensive and methodical, accommodating both casual users and developers.'),
        "structured_evaluation": StructuredEvaluationInstallationResult(
            install_available=True, 
            install_tutorial=True, 
            dependency_number=15, 
            dependency_suggestions='Include a note on verifying package manager availability (e.g., pip) before installation to avoid issues.', 
            compatible_os=False, overall_score='Good'
        ),
    }
    install_files = ['README.md', 'setup.py', 'requirements.txt']
    readme_evaluation = { "README.md": {
        "evaluation": {
            'project_level': True, 
            'score': 'Good', 
            'key_strengths': 'The README provides a clear introduction to the project and its purpose, detailed installation instructions, and robust usage examples that cater to varying user needs. The inclusion of API usage, integration examples, and specific domains of application is a significant strength, showcasing practical utility.', 
            'overall_improvement_suggestions': ['Hypotheses are central to information acquisition... discovery. - Consider condensing and adding specific use cases for immediate reader engagement.', 'Datasets will be automatically downloaded to specified data folder… - Explain flexibility for customizing paths or handling dataset-related errors.', 'A demo is provided in [here](demo.ipynb) to show how… - Include a snippet or screenshot to highlight expected demo functionality.', 'For any questions, please raise an issue… - Add explicit contributing guidelines detailing community participation methods.', 'Link to Paper… - Include a license section clarifying usage rights.']},
        "structured_evaluation": StructuredEvaluationREADMEResult(
            available_score=True, 
            readability_score='Good', 
            readability_suggestions='Simplify highly technical terms, provide layman-friendly definitions, or define jargon upfront. Consider adding a glossary for key terms.', 
            project_purpose_score=True, 
            project_purpose_suggestions='Add an explicit bullet-point outline listing objectives to improve clarity further.', 
            hardware_and_software_spec_score='Fair', 
            hardware_and_software_spec_suggestions='Clearly outline hardware requirements (e.g., minimum RAM, GPU specs) for locally-hosted benchmarks and APIs.', 
            dependency_score='Fair', 
            dependency_suggestions='Provide an explicit list of all dependencies (e.g., individual library names and versions) in the README alongside the requirements file.', 
            license_score=False, 
            license_suggestions='Add a LICENSE section to the README indicating the type of license applicable to the project and ensure presence of a LICENSE file.', 
            contributor_author_score=True, 
            overall_score='Fair'
        ),
    }}
    submit_evaluation, submit_files = mgr.evaluate_submission_requirements(
        readme_files_evaluation=readme_evaluation,
        installation_evaluation=install_evaluation,
        installation_files=install_files,
    )

    assert len(submit_files) > 0

