"""Unit tests for the CLI-invocation matching added to ConsistencyQueryStep.

No LLM is involved: the query step only reads the CliArgument table.  We seed a
small table with ``CodeStructureDb.insert_cli_argument`` and then check both the
pure ``_match_cli_invocation`` helper and the step's ``_execute_directly``.
"""

from bioguider.agents.consistency_query_step import (
    ConsistencyQueryStep,
    _match_cli_invocation,
)
from bioguider.agents.consistency_evaluation_task_utils import ConsistencyEvaluationState
from bioguider.database.code_structure_db import CodeStructureDb


def _seed_db(tmp_path):
    db = CodeStructureDb(author="acme", repo_name="tool", data_folder=str(tmp_path))
    # scripts/train.py: --epochs (int), --lr (float), positional `data`
    db.insert_cli_argument(path="scripts/train.py", dest="epochs",
                           option_strings=["--epochs"], prog="train", arg_type="int",
                           default_value="10")
    db.insert_cli_argument(path="scripts/train.py", dest="lr",
                           option_strings=["--lr"], prog="train", arg_type="float",
                           default_value="0.001")
    db.insert_cli_argument(path="scripts/train.py", dest="data",
                           option_strings=[], prog="train")
    # a sub-command parser: `mytool serve --port`
    db.insert_cli_argument(path="cli.py", dest="port", option_strings=["--port"],
                           prog="mytool", subcommand="serve", arg_type="int")
    return db


def _all_rows(db):
    return db.select_all_cli_arguments()


def test_match_known_python_invocation(tmp_path):
    db = _seed_db(tmp_path)
    inv = {
        "program": "python scripts/train.py",
        "language": "python",
        "subcommand": "N/A",
        "options": [
            {"name": "--epochs", "value": "20", "kind": "option"},
            {"name": "--lr", "value": "0.01", "kind": "option"},
            {"name": "<positional>", "value": "data.h5", "kind": "positional"},
        ],
        "source": "python scripts/train.py --epochs 20 --lr 0.01 data.h5",
    }
    f = _match_cli_invocation(_all_rows(db), inv)
    assert f["kind"] == "cli"
    assert f["status"] == "ok"
    assert f["resolved_paths"] == ["scripts/train.py"]
    assert sorted(f["matched_options"]) == ["--epochs", "--lr"]
    assert f["unknown_options"] == []
    assert f["issues"] == []


def test_match_unknown_option_is_flagged(tmp_path):
    db = _seed_db(tmp_path)
    inv = {
        "program": "scripts/train.py",
        "language": "python",
        "options": [
            {"name": "--epochs", "value": "5", "kind": "option"},
            {"name": "--epoch", "value": "5", "kind": "option"},   # typo
            {"name": "--workers", "value": "4", "kind": "option"},  # not defined
        ],
        "source": "python scripts/train.py --epochs 5 --epoch 5 --workers 4",
    }
    f = _match_cli_invocation(_all_rows(db), inv)
    assert f["status"] == "issues"
    assert f["matched_options"] == ["--epochs"]
    assert sorted(f["unknown_options"]) == ["--epoch", "--workers"]
    assert any("--epoch" in msg for msg in f["issues"])
    assert any("--workers" in msg for msg in f["issues"])


def test_unknown_program(tmp_path):
    db = _seed_db(tmp_path)
    inv = {"program": "python other_tool.py", "language": "python",
           "options": [{"name": "--foo", "value": "N/A", "kind": "flag"}],
           "source": "python other_tool.py --foo"}
    f = _match_cli_invocation(_all_rows(db), inv)
    assert f["status"] == "program_not_found"
    assert f["resolved_paths"] == []
    assert f["issues"] and "other_tool.py" in f["issues"][0]


def test_match_by_prog_and_subcommand(tmp_path):
    db = _seed_db(tmp_path)
    ok = _match_cli_invocation(_all_rows(db), {
        "program": "mytool", "language": "python", "subcommand": "serve",
        "options": [{"name": "--port", "value": "8080", "kind": "option"}],
        "source": "mytool serve --port 8080",
    })
    assert ok["status"] == "ok"
    assert ok["matched_options"] == ["--port"]

    bad = _match_cli_invocation(_all_rows(db), {
        "program": "mytool", "language": "python", "subcommand": "deploy",
        "options": [{"name": "--port", "value": "8080", "kind": "option"}],
        "source": "mytool deploy --port 8080",
    })
    assert bad["status"] == "issues"
    assert any("deploy" in msg for msg in bad["issues"])


def test_r_invocation_is_not_checked(tmp_path):
    db = _seed_db(tmp_path)
    f = _match_cli_invocation(_all_rows(db), {
        "program": "Rscript bin/analyze.R", "language": "r",
        "options": [{"name": "--input", "value": "x.csv", "kind": "option"}],
        "source": "Rscript bin/analyze.R --input x.csv",
    })
    assert f["status"] == "language_not_indexed"
    assert f["language"] == "r"
    # detected as R even when the collector forgot to set language
    f2 = _match_cli_invocation(_all_rows(db), {
        "program": "bin/analyze.R", "language": "unknown",
        "options": [], "source": "Rscript bin/analyze.R",
    })
    assert f2["language"] == "r"
    assert f2["status"] == "language_not_indexed"


def test_query_step_populates_cli_query_rows(tmp_path):
    db = _seed_db(tmp_path)
    state = ConsistencyEvaluationState(
        domain="user guide/API documentation",
        documentation="irrelevant for this test",
        step_output_callback=None,
        functions_and_classes=[],
        cli_invocations=[
            {"program": "python scripts/train.py", "language": "python",
             "options": [{"name": "--epochs", "value": "20", "kind": "option"}],
             "source": "python scripts/train.py --epochs 20"},
            {"program": "scripts/train.py", "language": "python",
             "options": [{"name": "--bogus", "value": "1", "kind": "option"}],
             "source": "python scripts/train.py --bogus 1"},
            {"program": "Rscript bin/analyze.R", "language": "r",
             "options": [], "source": "Rscript bin/analyze.R"},
        ],
    )
    step = ConsistencyQueryStep(code_structure_db=db)
    state, _token = step._execute_directly(state)

    assert state["all_query_rows"] == []          # no functions/classes given
    rows = state["cli_query_rows"]
    assert [r["status"] for r in rows] == ["ok", "issues", "language_not_indexed"]
    assert rows[1]["unknown_options"] == ["--bogus"]


def test_query_step_no_cli_invocations(tmp_path):
    db = _seed_db(tmp_path)
    state = ConsistencyEvaluationState(
        domain="d", documentation="x", step_output_callback=None,
        functions_and_classes=[],
    )
    step = ConsistencyQueryStep(code_structure_db=db)
    state, _token = step._execute_directly(state)
    assert state["cli_query_rows"] == []
