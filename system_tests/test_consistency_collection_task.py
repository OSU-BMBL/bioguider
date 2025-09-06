import pytest

from bioguider.agents.consistency_collection_task import ConsistencyCollectionTask
from bioguider.database.code_structure_db import CodeStructureDb
from bioguider.utils.code_structure_builder import CodeStructureBuilder

def test_consistency_collection_task(llm, step_callback, data_folder):
    with open("./system_tests/test_data/file_handler.md", "r") as f:
        user_guide_api_documentation = f.read()
    code_structure_db = CodeStructureDb(
        author="test",
        repo_name="test",
        data_folder=data_folder
    )
    code_structure_builder = CodeStructureBuilder(
        repo_path="./data/.adalflow/repos/OpenBMB_RepoAgent",
        gitignore_path="./data/.adalflow/repos/OpenBMB_RepoAgent/.gitignore",
        code_structure_db=code_structure_db,
    )
    # code_structure_builder.build_code_structure()
    
    task = ConsistencyCollectionTask(
        llm=llm,
        code_structure_db=code_structure_db,
        step_callback=step_callback,
    )
    task.compile(repo_path="./data/.adalflow/repos/OpenBMB_RepoAgent", gitignore_path="./data/.adalflow/repos/OpenBMB_RepoAgent/.gitignore")
    s = task.collect(user_guide_api_documentation=user_guide_api_documentation)
    assert s is not None