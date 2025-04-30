
from langchain.agents import Tool
from langchain.prompts import ChatPromptTemplate

from bioguider.agents.agent_tools import (
    read_directory_tool,
    read_file_tool,
    summarize_file_tool,
)
from bioguider.agents.agent_utils import (
    generate_repo_structure_prompt, 
    read_directory,
)
from bioguider.agents.collection_task_utils import (
    check_file_related_tool,
    RELATED_FILE_GOAL_ITEM,
)
from bioguider.agents.collection_execute_step import (
    CollectionExecuteStep,
    CollectionWorkflowState,
)
from bioguider.agents.python_ast_repl_tool import CustomPythonAstREPLTool
from bioguider.agents.prompt_utils import COLLECTION_PROMPTS, CollectionGoalItemEnum

def test_collection_execute_step(
    llm, 
    step_callback,
    plan_actions,
):
    repo_path = "/home/ubuntu/projects/github/scanpy"
    gitignore_path = "/home/ubuntu/projects/github/scanpy/.gitignore"
    files = read_directory(repo_path, gitignore_path)
    repo_structure = generate_repo_structure_prompt(files)
    goal_item_enum = CollectionGoalItemEnum.UserGuide.name
    related_file_goal_item_desc = ChatPromptTemplate.from_template(RELATED_FILE_GOAL_ITEM).format(
        goal_item=COLLECTION_PROMPTS[goal_item_enum]["goal_item"],
        related_file_description=COLLECTION_PROMPTS[goal_item_enum]["related_file_description"],
    )

    tools = [
        read_directory_tool(repo_path=repo_path),
        summarize_file_tool(
            llm=llm,
            repo_path=repo_path,
            token_usage_callback=step_callback,
        ),
        check_file_related_tool(
            llm=llm,
            repo_path=repo_path,
            goal_item_desc=related_file_goal_item_desc,
            token_usage_callback=step_callback,
        ),
        read_file_tool(repo_path=repo_path),
    ]
    custom_tools = [Tool(
        name=tool.__class__.__name__,
        func=tool.run,
        description=tool.__class__.__doc__,
    ) for tool in tools]
    custom_tools.append(CustomPythonAstREPLTool())

    step = CollectionExecuteStep(
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
        plan_actions=plan_actions,
    )
    state = step.execute(state)
    assert state is not None
    assert state["step_output"] is not None