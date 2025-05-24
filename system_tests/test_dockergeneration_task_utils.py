import pytest
import os
from langchain.tools import Tool
from bioguider.agents.dockergeneration_task_utils import generate_Dockerfile_tool, prepare_provided_files_string
from bioguider.agents.agent_utils import (
    read_directory,
    read_file,
    generate_repo_structure_prompt,
)
from bioguider.agents.dockergeneration_task_utils import (
    write_file_tool,
    generate_Dockerfile_tool,
)
from bioguider.agents.python_ast_repl_tool import CustomPythonAstREPLTool
from bioguider.utils.file_utils import extract_code_from_notebook

def test_generate_Dockerfile_tool(llm, step_callback):
    repo_path = "/home/ubuntu/projects/github/Slide_recon"
    gitignore_path = "/home/ubuntu/projects/github/Slide_recon/.gitignore"
    files = read_directory(repo_path, gitignore_path)
    repo_structure = generate_repo_structure_prompt(files, repo_path)
    provided_files = [
        "README.md", "environment.yml", "demo/example.ipynb",
        "scripts/qsub_seq_blind.sh", "scripts/reconstruction.sh",
    ]
    str_provided_files = prepare_provided_files_string(repo_path, provided_files)
    tools = [
        write_file_tool(repo_path),
        generate_Dockerfile_tool(
            llm, 
            repo_path, 
            str_provided_files, 
            repo_structure,
            step_callback,
        )
    ]
    custom_tools = [Tool(
        name=tool.__class__.__name__,
        func=tool.run,
        description=tool.__class__.__doc__,
    ) for tool in tools]
    custom_tools.append(CustomPythonAstREPLTool())

    # step = Docker

    tool = generate_Dockerfile_tool(
        llm, 
        repo_path, 
        str_provided_files, 
        repo_structure, 
        step_callback
    )
    dockerfile = tool.run()
    assert dockerfile is not None



