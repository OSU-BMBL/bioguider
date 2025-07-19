import pytest

from bioguider.agents.collection_task import CollectionTask
from bioguider.agents.collection_task_utils import CollectionWorkflowState
from bioguider.agents.prompt_utils import CollectionGoalItemEnum
from bioguider.database.summarized_file_db import SummarizedFilesDb

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

@pytest.mark.skip()
def test_collection_task_1(llm, step_callback):
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

@pytest.mark.skip()
def test_collection_task_for_installation_instructions(llm, step_callback):
    repo_path = "./data/repos/POPPER"
    gitignore_path = "./data/repos/POPPER/.gitignore"
    
    task = CollectionTask(
        llm=llm,
        step_callback=step_callback,
    )
    
    db = SummarizedFilesDb(
        author="snap-stanford",
        repo_name="POPPER",
    )
    task.compile(
        repo_path=repo_path, 
        gitignore_path=gitignore_path, 
        goal_item=CollectionGoalItemEnum.Installation.name,
        db=db,
    )
    s = task.collect()
    assert s is not None

def test_collection_task_for_installation(llm, step_callback, root_path):
    repo_path = f"{root_path}/POPPER"
    gitignore_path = f"{root_path}/POPPER/.gitignore"
    
    task = CollectionTask(
        llm=llm,
        step_callback=step_callback,
    )
    
    db = SummarizedFilesDb(
        author="snap-stanford",
        repo_name="POPPER",
    )
    task.compile(
        repo_path=repo_path, 
        gitignore_path=gitignore_path, 
        goal_item=CollectionGoalItemEnum.Installation.name,
        db=db,
    )
    results = task.collect()
    assert results is not None
    assert isinstance(results, list)
    
