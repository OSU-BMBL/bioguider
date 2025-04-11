
import pytest

from bioguider.agents.agent_utils import read_file_or_dir

def test_read_file_or_dir(llm):
    files = read_file_or_dir(
        name="/home/ubuntu/projects/github/tabula-data",
        repo_path="/home/ubuntu/projects/github/tabula-data",
        gitignore_path="/home/ubuntu/projects/github/tabula-data/.gitignore",
        llm=llm,
    )
    print(files)