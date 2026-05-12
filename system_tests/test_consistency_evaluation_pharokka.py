"""System test: run ConsistencyEvaluationTask over selected docs of the real
``pharokka`` repository (https://github.com/gbouras13/pharokka).

The module-scoped ``pharokka_db`` fixture does the one-time preparation —
clone the repo via RAG and build a CodeStructureDb (functions/classes + the
argparse CliArgument table) from it.  Each parametrized test then evaluates one
documentation page against that DB.

This clones a real repo and calls a real LLM, so it is slow and costs tokens —
run it deliberately, e.g.::

    pytest system_tests/test_consistency_evaluation_pharokka.py -s
"""

import logging
import os
import shutil
from pathlib import Path

import pytest

from bioguider.agents.consistency_evaluation_task import (
    ConsistencyEvaluationResult,
    ConsistencyEvaluationTask,
)
from bioguider.database.code_structure_db import CodeStructureDb
from bioguider.rag.rag import RAG
from bioguider.utils.code_structure_builder import CodeStructureBuilder

logger = logging.getLogger(__name__)

PHAROKKA_REPO_URL = "https://github.com/gbouras13/pharokka"

DOC_FILES = [
    # "docs/plotting.md",
    "docs/proteins.md",
    "docs/reasons.md",
    "docs/run.md",
]


@pytest.fixture(scope="module", autouse=True)
def cleanup_after_tests(data_folder):
    """Remove the sqlite databases created for this module after it finishes."""
    yield
    db_path = os.path.join(data_folder, "databases")
    if os.path.exists(db_path):
        logger.info("Cleaning up database directory: %s", db_path)
        try:
            shutil.rmtree(db_path)
        except OSError as e:  # pragma: no cover - best effort cleanup
            logger.warning("Could not clean up database directory: %s", e)


@pytest.fixture(scope="module")
def pharokka_db(data_folder):
    """Clone pharokka and build its CodeStructureDb once for the whole module."""
    rag = RAG()
    rag.initialize_db_manager()
    rag.initialize_repo(repo_url_or_path=PHAROKKA_REPO_URL)

    gitignore_path = Path(rag.repo_dir, ".gitignore")
    if not gitignore_path.exists():
        gitignore_path.write_text("", encoding="utf-8")

    code_structure_db = CodeStructureDb("pharokka_test", "pharokka_test", data_folder)
    builder = CodeStructureBuilder(
        repo_path=rag.repo_dir,
        gitignore_path=gitignore_path,
        code_structure_db=code_structure_db,
    )
    builder.build_code_structure()

    # sanity: something was indexed
    assert len(code_structure_db.select_all_names()) > 0, "pharokka code structure DB is empty"
    logger.info(
        "pharokka DB built: %d code symbols, %d CLI arguments",
        len(code_structure_db.select_all_names()),
        len(code_structure_db.select_all_cli_arguments()),
    )
    return code_structure_db, rag.repo_dir


@pytest.mark.parametrize("doc_rel_path", DOC_FILES)
def test_consistency_evaluation_pharokka_docs(llm, step_callback, pharokka_db, doc_rel_path):
    code_structure_db, repo_dir = pharokka_db
    doc_path = Path(repo_dir, doc_rel_path)
    if not doc_path.exists():
        pytest.skip(f"{doc_rel_path} is not present in the cloned pharokka repo")
    documentation = doc_path.read_text(encoding="utf-8", errors="ignore")

    task = ConsistencyEvaluationTask(
        llm=llm,
        code_structure_db=code_structure_db,
        step_callback=step_callback,
    )
    result = task.evaluate(
        domain="user guide/API documentation",
        documentation=documentation,
    )

    assert isinstance(result, ConsistencyEvaluationResult)
    assert 0 <= result.score <= 100
    assert isinstance(result.assessment, str) and result.assessment.strip()
    assert isinstance(result.development, list)
    assert isinstance(result.strengths, list)

    logger.info("[%s] score=%s", doc_rel_path, result.score)
    logger.info("[%s] assessment=%s", doc_rel_path, result.assessment)
    logger.info("[%s] development=%s", doc_rel_path, result.development)
    logger.info("[%s] strengths=%s", doc_rel_path, result.strengths)
