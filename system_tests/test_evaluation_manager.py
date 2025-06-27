
import pytest

from bioguider.managers.evaluation_manager import EvaluationManager
from bioguider.utils.constants import PrimaryLanguageEnum, ProjectTypeEnum

@pytest.mark.skip()
def test_EvaluationManager(llm, step_callback):
    mgr = EvaluationManager(llm, step_callback)
    mgr.prepare_repo("https://github.com/snap-stanford/POPPER")
    proj_type, language, meta_data = mgr.identify_project()
    assert proj_type == ProjectTypeEnum.package
    assert language == PrimaryLanguageEnum.python
    assert meta_data is not None and "name" in meta_data

    mgr.evaluate_readme()

def test_EvaluationManager_on_tutorial(llm, step_callback):
    mgr = EvaluationManager(llm, step_callback)
    mgr.prepare_repo("https://github.com/snap-stanford/POPPER")
    proj_metadata = mgr.identify_project()
    assert proj_metadata.project_type == ProjectTypeEnum.package
    assert proj_metadata.primary_language == PrimaryLanguageEnum.python
    assert proj_metadata.repo_name is not None

    mgr.evaluate_tutorial()

