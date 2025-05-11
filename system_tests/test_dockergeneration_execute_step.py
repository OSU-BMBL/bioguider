
import pytest
import os
from langchain.agents import Tool
from langchain.tools import StructuredTool
from langchain.prompts import ChatPromptTemplate

from bioguider.agents.dockergeneration_task_utils import (
    extract_python_file_from_notebook_tool,
    prepare_provided_files_string,
    write_file_tool,
    generate_Dockerfile_tool,
)
from bioguider.agents.agent_utils import (
    read_directory,
    read_file,
    generate_repo_structure_prompt,
)
from bioguider.agents.dockergeneration_execute_step import (
    DockerGenerationExecuteStep,
    DockerGenerationWorkflowState,
)
from bioguider.agents.python_ast_repl_tool import CustomPythonAstREPLTool
from bioguider.utils.file_utils import extract_code_from_notebook

PLAN_THOUGHTS = """### Reasoning Process\n\nThe current task is to prepare a working Dockerfile to run the demo example from the given repository. The intermediate Dockerfile failed due to a missing kernel (`slidelock`). To resolve this issue and ensure full reproducibility, the following steps are necessary:\n\n**Step 1**: Identify the root cause of the missing kernel issue.  \nThe error indicates that the kernel `slidelock` is unavailable during execution. This kernel name is specific to the environment of the author of the repository. We need to either install the kernel explicitly or decouple the code from this kernel dependency by running the Python script directly.\n\n**Step 2**: Extract notebook code into a Python script for easier execution.  \nPython notebooks are by default executed using a Jupyter environment, and they rely on kernel specifications. To avoid errors similar to the missing kernel issue (`slidelock`), we should extract the code in the notebook to a Python file. This allows us to run the demo independently using the configured environment.\n\n**Step 3**: Update the `environment.yml` file if required.  \nWe should ensure that the environment.yml file has all necessary dependencies installed for the demo example (e.g., UMAP, pandas, matplotlib, seaborn, etc.). These dependencies appear present based on provided data, so no action may be needed here.\n\n**Step 4**: Generate a new Dockerfile.  \nThe Dockerfile will copy the repository files, install the environment, and execute the extracted Python script from the notebook (`example.py`) instead of running the notebook directly.\n\n---\n\n### Final Plan Based on Reasoning\n\nTo address the kernel issue and meet reproducibility requirements:\n1. Extract the code from `demo/example.ipynb` into a Python script (`demo/example.py`) using the `extract_python_file_from_notebook_tool`.\n2. Generate a new Dockerfile using `generate_Dockerfile_tool`, ensuring it installs the required environment using `environment.yml` and executes the Python script (`example.py`).\n\n### Plan\n\nStep: extract_python_file_from_notebook_tool  \nStep Input: {"notebook_path": "demo/example.ipynb", "output_path": "demo/example.py"}\n\nStep: generate_Dockerfile_tool  \nStep Input: {"output_path": "demo-bioguider-5bojhlnyjc.Dockerfile"}
"""

def test_DockerGenerationExecuteStep(llm, step_callback):
    plan_actions = "Step: extract_python_file_from_notebook_tool\nStep Input: {'notebook_path': 'demo/example.ipynb', 'output_path': 'demo/example.py'}\nStep: generate_Dockerfile_tool\nStep Input: {'output_path': 'demo-bioguider-5bojhlnyjc.Dockerfile'}\n"
    plan_thoughts = PLAN_THOUGHTS
    repo_path = "/home/ubuntu/projects/github/Slide_recon"
    gitignore_path = "/home/ubuntu/projects/github/Slide_recon/.gitignore"
    files = read_directory(repo_path, gitignore_path)
    repo_structure = generate_repo_structure_prompt(files, repo_path)
    provided_files = [
        "README.md", "environment.yml", "demo/example.ipynb",
        "scripts/qsub_seq_blind.sh", "scripts/reconstruction.sh",
    ]
    str_provided_files = prepare_provided_files_string(repo_path, provided_files)
    write_tool = write_file_tool(repo_path)
    generate_tool = generate_Dockerfile_tool(
        llm=llm, 
        repo_path=repo_path, 
        extracted_files=str_provided_files, 
        repo_structure=repo_structure,
        output_callback=step_callback,
        plan_thoughts=plan_thoughts,
    )
    python_tool = CustomPythonAstREPLTool()
    extract_tool = extract_python_file_from_notebook_tool(repo_path)
    custom_tools = [
        StructuredTool.from_function(
            write_tool.run,
            description=write_tool.__class__.__doc__,
            name=write_tool.__class__.__name__,
        ),
        Tool(
            func=generate_tool.run,
            description=generate_tool.__class__.__doc__,
            name=generate_tool.__class__.__name__,
        ),
        python_tool,
        StructuredTool.from_function(
            extract_tool.run,
            description=extract_tool.__class__.__doc__,
            name=extract_tool.__class__.__name__,
        )
    ]

    step = DockerGenerationExecuteStep(
        llm=llm,
        repo_path=repo_path,
        repo_structure=repo_structure,
        gitignore_path=gitignore_path,
        custom_tools=custom_tools,
    )
    state = DockerGenerationWorkflowState(
        llm=llm,
        step_output_callback=step_callback,
        provided_files=provided_files,
        plan_actions=plan_actions,
        plan_thoughts=plan_thoughts,
    )
    state = step.execute(state)
    assert state is not None

