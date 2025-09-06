import os
import shutil
import pytest
from langchain.tools import Tool, StructuredTool
from pathlib import Path
from bioguider.agents.agent_tools import (
    read_directory_tool, 
    read_file_tool, 
    summarize_file_tool
)
from bioguider.agents.consistency_collection_task_utils import ConsistencyCollectionWorkflowState
from bioguider.agents.consistency_collection_plan_step import ConsistencyCollectionPlanStep
from bioguider.agents.consistency_collection_task_utils import (
    ConsistencyCollectionWorkflowState,
    retrieve_function_definition_and_docstring_tool,
    retrieve_class_definition_and_docstring_tool,
    retrieve_class_and_method_definition_and_docstring_tool,
)

from bioguider.database.code_structure_db import CodeStructureDb
from bioguider.utils.code_structure_builder import CodeStructureBuilder

@pytest.fixture(scope="module", autouse=True)
def cleanup_after_tests(data_folder):
    """Cleanup function that runs after all tests in this module complete."""
    # This will run before any tests in this module
    yield  # This is where the tests run
    
    # This will run after all tests in this module complete
    print("\n🧹 Running cleanup after all tests...")
    
    # Clean up database files
    db_path = os.path.join(data_folder, "databases")
    if os.path.exists(db_path):
        print(f"Cleaning up database directory: {db_path}")
        try:
            shutil.rmtree(db_path)
            print("✓ Database directory cleaned up")
        except Exception as e:
            print(f"⚠️  Warning: Could not clean up database directory: {e}")
    
    # Clean up any temporary files created during tests
    temp_files = [
        os.path.join(data_folder, "test_test.db"),
        os.path.join(data_folder, "databases", "test_test.db"),
    ]
    
    for temp_file in temp_files:
        if os.path.exists(temp_file):
            try:
                os.remove(temp_file)
                print(f"✓ Removed temporary file: {temp_file}")
            except Exception as e:
                print(f"⚠️  Warning: Could not remove {temp_file}: {e}")
    

def test_consistency_collection_plan_step(llm, step_callback, data_folder):
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
    with open("./system_tests/test_data/file_handler.md", "r") as f:
        user_guide_api_documentation = f.read()
    
    tools = [
        retrieve_function_definition_and_docstring_tool(llm=llm, code_structure_db=code_structure_db),
        retrieve_class_definition_and_docstring_tool(llm=llm, code_structure_db=code_structure_db),
        retrieve_class_and_method_definition_and_docstring_tool(llm=llm, code_structure_db=code_structure_db),
    ]
    custom_tools = [StructuredTool.from_function(
        tools[0].run,
        description=tools[0].__class__.__doc__,
        name=tools[0].__class__.__name__,
    ), StructuredTool.from_function(
        tools[1].run,
        description=tools[1].__class__.__doc__,
        name=tools[1].__class__.__name__,
    ), StructuredTool.from_function(
        tools[2].run,
        description=tools[2].__class__.__doc__,
        name=tools[2].__class__.__name__,
    )]
    step = ConsistencyCollectionPlanStep(
        llm=llm,
        custom_tools=custom_tools,
    )
    state = ConsistencyCollectionWorkflowState(
        intermediate_steps=[],
        user_guide_api_documentation=user_guide_api_documentation,
        step_output_callback=step_callback,
    )
    state = step.execute(state)
    assert state is not None
    assert state["plan_actions"] is not None
