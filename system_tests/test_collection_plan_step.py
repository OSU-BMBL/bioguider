import pytest
from langchain.tools import Tool
from bioguider.agents.agent_tools import (
    read_directory_tool, 
    read_file_tool, 
    summarize_file_tool
)
from bioguider.agents.collection_task_utils import (
    check_file_related_tool,
)
from bioguider.agents.collection_plan_step import (
    CollectionPlanStep,
    CollectionWorkflowState,
)
from bioguider.agents.agent_utils import (
    read_directory,
    generate_repo_structure_prompt
)
from bioguider.agents.python_ast_repl_tool import CustomPythonAstREPLTool
from bioguider.agents.prompt_utils import COLLECTION_PROMPTS

def test_collection_plan_step(llm, step_callback):
    repo_path = "/home/ubuntu/projects/github/tabula-data"
    gitignore_path = "/home/ubuntu/projects/github/tabula-data/.gitignore"
    files = read_directory(repo_path, gitignore_path)
    repo_structure = generate_repo_structure_prompt(files, repo_path)

    tools = [
        read_directory_tool(repo_path=repo_path),
        summarize_file_tool(
            llm=llm,
            repo_path=repo_path,
            output_callback=step_callback,
        ),
        check_file_related_tool(
            llm=llm,
            repo_path=repo_path,
            goal_item_desc=COLLECTION_PROMPTS["UserGuide"]["related_file_description"],
            output_callback=step_callback,
        ),
        read_file_tool(repo_path=repo_path),
    ]
    custom_tools = [Tool(
        name=tool.__class__.__name__,
        func=tool.run,
        description=tool.__class__.__doc__,
    ) for tool in tools]
    custom_tools.append(CustomPythonAstREPLTool())

    step = CollectionPlanStep(
        llm=llm,
        repo_path=repo_path,
        repo_structure=repo_structure,
        gitignore_path=gitignore_path,
        custom_tools=custom_tools,
    )
    state = CollectionWorkflowState(
        intermediate_steps=[],
        goal_item="installation",
        step_output_callback=step_callback,
    )
    state = step.execute(state)

    assert state is not None
    assert "plan_actions" in state and len(state["plan_actions"]) > 0

    
