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

@pytest.mark.skip()
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


def _inject_cores_into_multiplotter_line(plotting_md: str) -> str:
    """Append ``--cores 4`` to the ``pharokka_multiplotter.py`` invocation line.

    Mirrors the corruption produced by ``cli_unknown_flag`` in the benchmark
    injector, but stays deterministic so the test isn't coupled to the
    injector's choices.  Raises ``ValueError`` if the expected line is not
    found, so the test fails loudly when pharokka's doc structure shifts.
    """
    """
    new_lines: list[str] = []
    modified = False
    for line in plotting_md.splitlines(keepends=True):
        if (
            "pharokka_multiplotter.py" in line
            and "-g pharokka.gbk" in line
            and "--cores" not in line
        ):
            head = line.rstrip()
            if line.endswith("\r\n"):
                eol = "\r\n"
            elif line.endswith("\n"):
                eol = "\n"
            else:
                eol = ""
            new_lines.append(head + " --cores 4" + eol)
            modified = True
        else:
            new_lines.append(line)
    if not modified:
        raise ValueError(
            "could not find the pharokka_multiplotter.py invocation line "
            "in docs/plotting.md — has the upstream doc structure changed?"
        )
    return "".join(new_lines)
    """

    corrupted_file = Path(__file__).parent.parent / "outputs"/ "pipeline_stress" / "run_20260514_131724" / "plotting.level_10.corrupted.md"
    return corrupted_file.read_text(encoding="utf-8", errors="ignore")


def test_consistency_flags_multiplotter_cores_as_undefined(
    llm, step_callback, pharokka_db
):
    """The consistency task must flag ``--cores`` on ``pharokka_multiplotter.py``
    when evaluating the real ``docs/plotting.md``.

    Pharokka's ``pharokka_multiplotter.py`` parser defines ``-g/--genbank``
    and ``-o/--outdir`` (among others) but no ``--cores`` option.  We take
    the *real* user guide page, append ``--cores 4`` to the multiplotter
    invocation (the same shape ``cli_unknown_flag`` produces in the
    benchmark), and assert the consistency task surfaces it — the larger
    doc context exercises the task realistically rather than via a
    minimal hand-crafted snippet.
    """
    code_structure_db, repo_dir = pharokka_db

    plotting_path = Path(repo_dir) / "docs" / "plotting.md"
    if not plotting_path.exists():
        pytest.skip("docs/plotting.md not present in cloned pharokka repo")
    original_doc = plotting_path.read_text(encoding="utf-8", errors="ignore")

    # Preconditions on the upstream doc — the test only makes sense if
    # ``--cores`` is genuinely *not* already documented for the multiplotter.
    assert "--cores" not in original_doc, (
        "docs/plotting.md unexpectedly already contains '--cores'; the "
        "injected flaw would not be a real fault"
    )

    documentation = _inject_cores_into_multiplotter_line(original_doc)
    assert "--cores 4" in documentation
    # The injection should be local to one line — sanity that we didn't
    # accidentally splat the suffix everywhere.
    assert documentation.count("--cores") == 1

    # Sanity: pharokka_multiplotter.py is indexed and --cores is NOT defined.
    all_rows = code_structure_db.select_all_cli_arguments()
    mp_rows = [r for r in all_rows if r["path"].endswith("pharokka_multiplotter.py")]
    assert mp_rows, (
        "pharokka_multiplotter.py not indexed in the CliArgument table — "
        "indexing must have failed for this entry point"
    )
    mp_options = {opt for r in mp_rows for opt in (r.get("option_strings") or [])}
    assert "--cores" not in mp_options, (
        f"pharokka_multiplotter.py unexpectedly defines --cores: {sorted(mp_options)}"
    )
    assert any(opt in mp_options for opt in ("--genbank", "-g", "--outdir", "-o")), (
        f"none of the expected pharokka_multiplotter flags are indexed: {sorted(mp_options)}"
    )

    # Run the full consistency task on the real (mutated) plotting.md.
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

    logger.info("score=%s", result.score)
    logger.info("assessment=%s", result.assessment)
    logger.info("development=%s", result.development)
    logger.info("strengths=%s", result.strengths)

    haystack = " ".join([result.assessment, *result.development]).lower()
    assert "cores" in haystack, (
        "expected the evaluation to flag the undefined '--cores' flag for "
        "pharokka_multiplotter.py; got "
        f"development={result.development!r} assessment={result.assessment!r}"
    )
