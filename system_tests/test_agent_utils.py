
import pytest

from bioguider.agents.agent_utils import read_directory

def test_read_directory(llm):
    files = read_directory(
        dir_path="./bioguider",
        gitignore_path="./bioguider/.gitignore",
    )
    assert len(files) > 0