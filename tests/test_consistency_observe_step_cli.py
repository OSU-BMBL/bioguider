"""Unit tests for the CLI-findings rendering used by ConsistencyObserveStep.

No LLM is involved — only ``_format_cli_findings`` (pure text formatting of the
``cli_query_rows`` produced by ConsistencyQueryStep).
"""

from bioguider.agents.consistency_observe_step import _format_cli_findings


def test_empty_findings():
    assert "No command-line invocations" in _format_cli_findings(None)
    assert "No command-line invocations" in _format_cli_findings([])


def test_ok_finding_renders_program_and_matched_options():
    text = _format_cli_findings([
        {
            "kind": "cli",
            "program": "python scripts/train.py",
            "language": "python",
            "subcommand": None,
            "source": "python scripts/train.py --epochs 20",
            "status": "ok",
            "resolved_paths": ["scripts/train.py"],
            "defined_options": ["--epochs", "--lr"],
            "matched_options": ["--epochs"],
            "unknown_options": [],
            "issues": [],
        }
    ])
    assert "python scripts/train.py" in text
    assert "scripts/train.py" in text
    assert "--epochs" in text
    assert "status: ok" in text
    assert "consistent with the code" in text


def test_issues_finding_lists_unknown_options_and_messages():
    text = _format_cli_findings([
        {
            "kind": "cli",
            "program": "scripts/train.py",
            "language": "python",
            "subcommand": "train",
            "source": "python scripts/train.py train --workers 4",
            "status": "issues",
            "resolved_paths": ["scripts/train.py"],
            "defined_options": ["--epochs"],
            "matched_options": [],
            "unknown_options": ["--workers"],
            "issues": ["The documentation passes '--workers' to 'scripts/train.py', "
                       "but the parser defines no such option (known: ['--epochs'])."],
        }
    ])
    assert "INCONSISTENT" in text
    assert "sub-command used: train" in text
    assert "documented options the parser does NOT define: --workers" in text
    assert "issues:" in text
    assert "--workers" in text


def test_program_not_found_finding():
    text = _format_cli_findings([
        {"kind": "cli", "program": "python other.py", "language": "python",
         "subcommand": None, "source": "python other.py --foo", "status": "program_not_found",
         "resolved_paths": [], "defined_options": [], "matched_options": [],
         "unknown_options": [], "issues": ["...no command-line parser for it was found..."]},
    ])
    assert "NOT FOUND" in text


def test_language_not_indexed_finding_is_neutral():
    text = _format_cli_findings([
        {"kind": "cli", "program": "Rscript bin/analyze.R", "language": "r",
         "subcommand": None, "source": "Rscript bin/analyze.R --input x.csv",
         "status": "language_not_indexed", "resolved_paths": [], "defined_options": [],
         "matched_options": [], "unknown_options": [],
         "issues": ["R command-line options are not indexed yet; this invocation was not checked."]},
    ])
    assert "not checked" in text
    assert "Rscript bin/analyze.R" in text


def test_multiple_findings_are_separated():
    text = _format_cli_findings([
        {"kind": "cli", "program": "a.py", "language": "python", "subcommand": None,
         "source": "python a.py", "status": "ok", "resolved_paths": ["a.py"],
         "defined_options": [], "matched_options": [], "unknown_options": [], "issues": []},
        {"kind": "cli", "program": "b.py", "language": "python", "subcommand": None,
         "source": "python b.py --x", "status": "program_not_found", "resolved_paths": [],
         "defined_options": [], "matched_options": [], "unknown_options": [], "issues": ["nope"]},
    ])
    assert text.count("program (as written in the documentation):") == 2
