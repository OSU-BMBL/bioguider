import pytest

from bioguider.agents.collection_task_utils import check_file_related_tool
from bioguider.agents.prompt_utils import COLLECTION_PROMPTS

def test_check_file_related_tool(llm, step_callback):
    tool = check_file_related_tool(
        llm=llm,
        repo_path="/home/ubuntu/projects/github/biochatter",
        goal_item_desc=COLLECTION_PROMPTS["UserGuide"]["related_file_description"],
        output_callback=step_callback,
    )

    res = tool.run(file_path="README.md")
    assert res[:3] == "Yes"


