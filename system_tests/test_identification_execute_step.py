import pytest

from langchain.tools import Tool, StructuredTool

from bioguider.agents.agent_tools import read_directory_tool, read_file_tool, summarize_file_tool
from bioguider.agents.agent_utils import generate_repo_structure_prompt, read_directory
from bioguider.agents.identification_execute_step import (
    IdentificationExecuteStep,
)
from bioguider.agents.identification_task_utils import (
    IdentificationWorkflowState,
)
from bioguider.agents.prompt_utils import IDENTIFICATION_GOAL_PROJECT_TYPE
from bioguider.agents.python_ast_repl_tool import CustomPythonAstREPLTool

def test_identification_plan_step(llm, step_callback):
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
        read_file_tool(repo_path=repo_path),
    ]
    custom_tools = [Tool(
        name=tools[0].__class__.__name__,
        func=tools[0].run,
        description=tools[0].__class__.__doc__,
    ), StructuredTool.from_function(
        tools[1].run,
        description=tools[1].__class__.__doc__,
        name=tools[1].__class__.__name__,
    ), Tool(
        name=tools[2].__class__.__name__,
        func=tools[2].run,
        description=tools[2].__class__.__doc__,
    ),]
    custom_tools.append(CustomPythonAstREPLTool())

    step = IdentificationExecuteStep(
        llm=llm,
        repo_path=repo_path,
        repo_structure=repo_structure,
        gitignore_path=gitignore_path,
        custom_tools=custom_tools,
    )
    state = IdentificationWorkflowState(
        intermediate_steps=[],
        goal=IDENTIFICATION_GOAL_PROJECT_TYPE,
        step_output_callback=step_callback,
        plan_actions="Step: summarize_file_tool\nStep Input: pyproject.toml\n"
    )

    state = step.execute(state)

    assert state is not None
    assert "step_output" in state and len(state["step_output"]) > 0