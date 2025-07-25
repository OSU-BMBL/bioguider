
from langchain.tools import Tool
from langchain.prompts import ChatPromptTemplate

from bioguider.agents.agent_utils import generate_repo_structure_prompt, read_directory
from bioguider.agents.collection_observe_step import CollectionObserveStep
from bioguider.agents.collection_task_utils import (
    CollectionWorkflowState,
)
from bioguider.agents.prompt_utils import CollectionGoalItemEnum


def test_collection_observe_step(llm, step_callback):
    repo_path = "./data/repos/POPPER"
    gitignore_path = "./data/repos/POPPER/.gitignore"
    files = read_directory(repo_path, gitignore_path)
    repo_structure = generate_repo_structure_prompt(files, repo_path)

    step = CollectionObserveStep(
        llm=llm,
        repo_path=repo_path,
        repo_structure=repo_structure,
        gitignore_path=gitignore_path,
    )
    state = CollectionWorkflowState(
        intermediate_steps=[],
        goal_item=CollectionGoalItemEnum.UserGuide.name,
        step_output_callback=step_callback,
        step_output="README.md: Yes, the file **is** related to the goal item.\n"
    )
    state = step.execute(state)
    assert state is not None
    assert state["step_thoughts"] is not None
    

