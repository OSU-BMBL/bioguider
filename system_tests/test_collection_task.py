import pytest

from bioguider.agents.collection_task import CollectionTask
from bioguider.agents.collection_task_utils import CollectionWorkflowState
from bioguider.agents.prompt_utils import CollectionGoalItemEnum

@pytest.mark.skip()
def test_collection_task(llm, step_callback):
    repo_path = "/home/ubuntu/projects/github/seurat"
    gitignore_path = "/home/ubuntu/projects/github/seurat/.gitignore"
    
    task = CollectionTask(
        llm=llm,
        step_callback=step_callback,
    )
    
    task.compile(
        repo_path=repo_path, 
        gitignore_path=gitignore_path, 
        goal_item=CollectionGoalItemEnum.Tutorial.name,
    )
    s = task.collect()
    assert s is not None

def test_collection_task(llm, step_callback):
    repo_path = "/home/ubuntu/projects/github/Slide_recon"
    gitignore_path = "/home/ubuntu/projects/github/Slide_recon/.gitignore"
    
    task = CollectionTask(
        llm=llm,
        step_callback=step_callback,
    )
    
    task.compile(
        repo_path=repo_path, 
        gitignore_path=gitignore_path, 
        goal_item=CollectionGoalItemEnum.DockerGeneration.name,
    )
    s = task.collect()
    assert s is not None