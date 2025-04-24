
import pytest

# from bioguider.agents.collect_agent import CollectionAgent, IdentificationPlanResult

from bioguider.agents.identification_step import (
    IdentificationStep, 
    IdentificationPlanResult,
    ProjectTypeEnum,
    PrimaryLanguageEnum
)

# @pytest.mark.skip()
def test_IdentificationAgent_biochatter(llm, step_callback):
    json_schema = IdentificationPlanResult.model_json_schema()

    step = IdentificationStep(llm=llm, step_callback=step_callback)
    
    step.compile(
        repo_path="/home/ubuntu/projects/github/POPPER/",
        gitignore_path="/home/ubuntu/projects/github/POPPER/.gitignore",
    )
    res = step.identify_project_type()
    assert res == ProjectTypeEnum.package
    print(res)

@pytest.mark.skip()
def test_IdentificationAgent_biochatter_server(llm, step_callback):
    json_schema = IdentificationPlanResult.model_json_schema()

    step = IdentificationStep(llm=llm, step_callback=step_callback)
    
    step.compile(
        repo_path="/home/ubuntu/projects/github/biochatter-server/",
        gitignore_path="/home/ubuntu/projects/github/biochatter-server/.gitignore",
    )
    res = step.identify_project_type()
    assert res == ProjectTypeEnum.package
    print(res)

@pytest.mark.skip()
def test_IdentificationAgent_biochatter_server(llm, step_callback):
    json_schema = IdentificationPlanResult.model_json_schema()

    step = IdentificationStep(llm=llm, step_callback=step_callback)
    
    step.compile(
        repo_path="/home/ubuntu/projects/github/biochatter-server/",
        gitignore_path="/home/ubuntu/projects/github/biochatter-server/.gitignore",
    )
    res = step.identify_primary_language()
    assert res == PrimaryLanguageEnum.python
    print(res)



