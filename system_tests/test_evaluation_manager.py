
import pytest

from bioguider.managers.evaluation_manager import EvaluationManager
from bioguider.utils.constants import PrimaryLanguageEnum, ProjectTypeEnum
from bioguider.agents.evaluation_task import EvaluationREADMEResult

# @pytest.mark.skip()
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

@pytest.mark.skip()
def test_EvaluationManager_on_installation(llm, step_callback):
    json_obj = EvaluationREADMEResult.model_json_schema()
    mgr = EvaluationManager(llm, step_callback)
    mgr.prepare_repo("https://github.com/OSU-BMBL/deepmaps")

    evaluation, files = mgr.evaluate_installation()
    assert len(files) > 0