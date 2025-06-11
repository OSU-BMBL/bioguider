
import pytest

from bioguider.agents.identification_task import PrimaryLanguageEnum, ProjectTypeEnum
from bioguider.managers.evaluation_manager import EvaluationManager

def test_EvaluationManager(llm, step_callback):
    mgr = EvaluationManager(llm, step_callback)
    mgr.prepare_repo("https://github.com/snap-stanford/POPPER")
    proj_type, language, meta_data = mgr.identify_project()
    assert proj_type == ProjectTypeEnum.package
    assert language == PrimaryLanguageEnum.python
    assert meta_data is not None and "name" in meta_data

