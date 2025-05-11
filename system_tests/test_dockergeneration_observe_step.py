
import pytest
import os
from langchain.agents import Tool
from langchain.prompts import ChatPromptTemplate

from bioguider.agents.dockergeneration_task_utils import (
    write_file_tool,
    generate_Dockerfile_tool,
)
from bioguider.agents.agent_utils import (
    read_directory,
    read_file,
    generate_repo_structure_prompt,
)
from bioguider.agents.dockergeneration_observe_step import (
    DockerGenerationObserveStep,
    DockerGenerationWorkflowState,
)
from bioguider.agents.python_ast_repl_tool import CustomPythonAstREPLTool
from bioguider.utils.file_utils import extract_code_from_notebook

def test_DockerGenerationExecuteStep(llm, step_callback):
    dockerfile = "demo-bioguider-wvHOMqOxcZ.Dockerfile"
    repo_path = "/home/ubuntu/projects/github/Slide_recon"
    
    step = DockerGenerationObserveStep(
        llm=llm,
        repo_path=repo_path,
    )
    state = DockerGenerationWorkflowState(
        llm=llm,
        step_output_callback=step_callback,
        dockerfile=dockerfile,
    )
    state = step.execute(state)
    assert state is not None

