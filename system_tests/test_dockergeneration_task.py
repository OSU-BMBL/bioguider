import os
import pytest

from bioguider.agents.agent_utils import read_file
from bioguider.agents.dockergeneration_task import DockerGenerationTask
from bioguider.agents.dockergeneration_task_utils import prepare_provided_files_string

def test_DockerGenerationTask(llm, step_callback):
    repo_path = "/home/ubuntu/projects/github/Slide_recon"
    gitignore_path = "/home/ubuntu/projects/github/Slide_recon/.gitignore"

    provided_files = [
        "README.md", "environment.yml", "demo/example.ipynb",
        "scripts/qsub_seq_blind.sh", "scripts/reconstruction.sh",
    ]
    # str_provided_files = prepare_provided_files_string(repo_path, provided_files)

    task = DockerGenerationTask(
        llm=llm,
        step_callback=step_callback,
    )
    task.compile(
        repo_path=repo_path,
        gitignore_path=gitignore_path,
        provided_files=provided_files,
    )
    s = task.generate()
    assert s is not None



