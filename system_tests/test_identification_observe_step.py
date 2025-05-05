import pytest

from langchain.tools import Tool

from bioguider.agents.agent_utils import generate_repo_structure_prompt, read_directory
from bioguider.agents.identification_observe_step import IdentificationObserveStep
from bioguider.agents.identification_task_utils import IdentificationWorkflowState
from bioguider.agents.agent_tools import read_directory_tool, read_file_tool, summarize_file_tool
from bioguider.agents.python_ast_repl_tool import CustomPythonAstREPLTool
from bioguider.agents.prompt_utils import IDENTIFICATION_GOAL_PROJECT_TYPE

def test_identification_observe_step(llm, step_callback):
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
        name=tool.__class__.__name__,
        func=tool.run,
        description=tool.__class__.__doc__,
    ) for tool in tools]
    custom_tools.append(CustomPythonAstREPLTool())

    step = IdentificationObserveStep(
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
        step_output="Action: summarize_file_tool\nAction Input: pyproject.toml\nAction Observation: summarized content of file pyproject.toml: The file **`pyproject.toml`** specifies the configurations for a Python project named **`tabula-data`**, maintained by **fengsh**. It uses Poetry for dependency and build management, with a listed version **0.2.1** and an MIT license. The project primarily depends on Python **^3.10**, along with key libraries such as **pandas**, **requests**, **beautifulsoup4**, **streamlit**, **openai**, and others, to handle tasks like data manipulation, HTTP requests, and AI-driven functionalities. Two optional extras—**semantic** (for text embedding via **sentence-transformers**) and **claude** (for using Anthropic's API)—are defined.\n\nThe project also specifies development dependencies, including **pytest** for testing, **bump2version** for versioning, and **pre-commit** for Git hooks. It integrates multiple LangChain-related dependencies, such as **langchain**, **langchain-openai**, and community plugins, which suggest its focus on advanced AI or language model workflows. The configuration follows the **PEP 518** standard with `poetry-core` as its build backend. Overall, this file defines a rich ecosystem of tools for building AI-driven or data-related applications with support for extensibility and collaboration.\n---\n",
    )

    state = step.execute(state)

    assert state is not None