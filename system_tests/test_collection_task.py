

from bioguider.agents.collection_task import CollectionTask
from bioguider.agents.collection_task_utils import CollectionWorkflowState
from bioguider.agents.prompt_utils import CollectionGoalItemEnum

def test_collection_task(llm, step_callback):
    repo_path = "/home/ubuntu/projects/github/seurat"
    gitignore_path = "/home/ubuntu/projects/github/seurat/.gitignore"
    
    task = CollectionTask(
        llm=llm,
        step_callback=step_callback,
        goal_item=CollectionGoalItemEnum.Tutorial.name,
    )
    
    task.compile(
        repo_path=repo_path, 
        gitignore_path=gitignore_path, 
    )
    s = task.collect()
    assert s is not None
