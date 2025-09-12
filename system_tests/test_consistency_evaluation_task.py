
import os
from pathlib import Path
import shutil
import pytest
import logging
import json

from bioguider.agents.consistency_evaluation_task import ConsistencyEvaluationTask
from bioguider.database.code_structure_db import CodeStructureDb
from bioguider.rag.rag import RAG
from bioguider.utils.code_structure_builder import CodeStructureBuilder
from bioguider.agents.consistency_collection_step import ConsistencyCollectionResult

logger = logging.getLogger(__name__)

@pytest.fixture(scope="module", autouse=True)
def cleanup_after_tests(data_folder):
    
    """Cleanup function that runs after all tests in this module complete."""
    # This will run before any tests in this module
    yield  # This is where the tests run
            
    # Clean up database files
    db_path = os.path.join(data_folder, "databases")
    if os.path.exists(db_path):
        logger.info(f"Cleaning up database directory: {db_path}")
        try:
            shutil.rmtree(db_path)
            logger.info("Database directory cleaned up")
        except Exception as e:
            logger.warning(f"Warning: Could not clean up database directory: {e}")

def test_ConsistencyEvaluationTask(llm, step_callback, data_folder):
    schema = ConsistencyCollectionResult.model_json_schema()
    logger.info(json.dumps(schema, indent=2))

    # prepare database
    repo_url = "https://github.com/OpenBMB/RepoAgent"
    rag = RAG()
    rag.initialize_db_manager()
    rag.initialize_repo(repo_url_or_path=repo_url)
    
    # summary_file_db = SummarizedFilesDb(author, repo_name)
    code_structure_db = CodeStructureDb(
        "test", "test", data_folder
    )
    code_structure_builder = CodeStructureBuilder(
        repo_path=rag.repo_dir, 
        gitignore_path=Path(rag.repo_dir, ".gitignore"), 
        code_structure_db=code_structure_db
    )
    code_structure_builder.build_code_structure()

    # test
    with open(Path(rag.repo_dir, "markdown_docs/repo_agent/file_handler.md"), "r") as f:
        user_guide_api_documentation = f.read()
        
    task = ConsistencyEvaluationTask(
        llm=llm,
        code_structure_db=code_structure_db,
        step_callback=step_callback,
    )
    state = task.evaluate(user_guide_api_documentation=user_guide_api_documentation)
    assert state is not None