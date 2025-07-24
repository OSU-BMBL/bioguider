
from pydantic import BaseModel, Field
import pytest

from bioguider.agents.agent_utils import read_directory, try_parse_with_llm

def test_read_directory(llm):
    files = read_directory(
        dir_path="./bioguider",
        gitignore_path="./bioguider/.gitignore",
    )
    assert len(files) > 0

class TestSchema(BaseModel):
    name: str = Field(description="The name of the person")
    age: int = Field(description="The age of the person")

def test_try_parse_with_llm(llm):
    result = try_parse_with_llm(llm, '{"name": "John", "age": 30}', TestSchema)
    assert result is not None