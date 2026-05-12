"""End-to-end system test: ConsistencyEvaluationTask should flag a command-line
flag that appears in the documentation but is not defined by the code's argparse
parser.  This exercises the full chain — collection (cli_invocations) → query
(cli_query_rows, matched against the CliArgument table) → observe (renders the
findings into the prompt).  It calls a real LLM, so it is a smoke test, but the
assertion on the bogus flag is reasonable: the query step hands the model an
explicit "the parser does NOT define --workers" note.
"""

import logging
import os
import shutil
import textwrap
from pathlib import Path

import pytest

from bioguider.agents.consistency_evaluation_task import (
    ConsistencyEvaluationResult,
    ConsistencyEvaluationTask,
)
from bioguider.database.code_structure_db import CodeStructureDb
from bioguider.utils.code_structure_builder import CodeStructureBuilder

logger = logging.getLogger(__name__)


# `--workers` is intentionally NOT a flag the parser defines; `--epochs` is.
CLI_DOCUMENTATION = textwrap.dedent("""\
    ## Training

    Train a model from the command line:

    ```bash
    python train.py --epochs 50 --workers 8 data/train.h5
    ```

    The positional argument is the path to the input dataset.
    """)


def _make_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "cli_repo"
    repo.mkdir()
    (repo / ".gitignore").write_text("", encoding="utf-8")
    (repo / "train.py").write_text(textwrap.dedent('''
        """Train a model."""
        import argparse


        def main():
            parser = argparse.ArgumentParser(prog="train", description="Train a model.")
            parser.add_argument("--epochs", type=int, default=10,
                                help="number of training epochs")
            parser.add_argument("--lr", type=float, default=0.001,
                                help="learning rate")
            parser.add_argument("data", help="path to the input dataset")
            return parser.parse_args()


        if __name__ == "__main__":
            main()
    '''), encoding="utf-8")
    return repo


@pytest.fixture(scope="module")
def cli_db(data_folder, tmp_path_factory):
    repo = _make_repo(tmp_path_factory.mktemp("cli"))
    db = CodeStructureDb("cli_consistency_test", "cli_consistency_test", data_folder)
    builder = CodeStructureBuilder(
        repo_path=repo,
        gitignore_path=Path(repo, ".gitignore"),
        code_structure_db=db,
    )
    builder.build_code_structure()
    # sanity: argparse options were indexed into the CliArgument table
    rows = db.select_cli_arguments_by_path("train.py")
    dests = {r["dest"] for r in rows}
    assert {"epochs", "lr", "data"} <= dests, rows
    yield db
    db_dir = os.path.join(data_folder, "databases")
    if os.path.exists(db_dir):
        try:
            shutil.rmtree(db_dir)
        except OSError as e:  # pragma: no cover - best effort cleanup
            logger.warning("Could not clean up %s: %s", db_dir, e)


def test_consistency_flags_undefined_cli_option(llm, step_callback, cli_db):
    task = ConsistencyEvaluationTask(
        llm=llm,
        code_structure_db=cli_db,
        step_callback=step_callback,
    )
    result = task.evaluate(
        domain="user guide/API documentation",
        documentation=CLI_DOCUMENTATION,
    )

    assert isinstance(result, ConsistencyEvaluationResult)
    assert 0 <= result.score <= 100
    assert isinstance(result.development, list)

    logger.info("score=%s", result.score)
    logger.info("assessment=%s", result.assessment)
    logger.info("development=%s", result.development)
    logger.info("strengths=%s", result.strengths)

    haystack = " ".join([result.assessment, *result.development]).lower()
    assert "workers" in haystack, (
        "expected the evaluation to call out the undefined '--workers' flag; "
        f"got development={result.development!r} assessment={result.assessment!r}"
    )
