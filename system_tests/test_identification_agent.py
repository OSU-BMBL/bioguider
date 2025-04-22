
import pytest

# from bioguider.agents.collect_agent import CollectionAgent, IdentificationPlanResult

from bioguider.agents.identification_step import IdentificationStep, IdentificationPlanResult

@pytest.mark.skip()
def test_CollectionAgent_biochatter(llm, step_callback):
    json_schema = IdentificationPlanResult.model_json_schema()

    step = IdentificationStep(llm=llm, step_callback=step_callback)
    
    step.compile(
        repo_path="/home/ubuntu/projects/github/POPPER/",
        gitignore_path="/home/ubuntu/projects/github/POPPER/.gitignore",
    )
    res = step.execute()
    assert res == "package"
    print(res)

def test_CollectionAgent_tabula_data(llm, step_callback):
    json_schema = IdentificationPlanResult.model_json_schema()

    step = IdentificationStep(llm=llm, step_callback=step_callback)
    
    step.compile(
        repo_path="/home/ubuntu/projects/github/NetToolkit/",
        gitignore_path="/home/ubuntu/projects/github/NetToolkit/.gitignore",
    )
    res = step.execute()
    assert res == "package"
    print(res)


