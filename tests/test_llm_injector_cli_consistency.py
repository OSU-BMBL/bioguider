"""Unit tests for CLI-consistency error injection.

Covers the three deterministic categories ``cli_flag_typo``,
``cli_unknown_flag``, ``cli_program_rename`` added to ``LLMErrorInjector``.
No LLM is called — every test uses ``force_deterministic=True`` or calls the
private helpers directly.
"""
import pytest

from bioguider.generation.llm_injector import (
    LLMErrorInjector,
    _CLI_BOGUS_FLAGS,
    _CODE_FENCE_RE,
)


def _make_injector():
    return LLMErrorInjector(llm=None, force_deterministic=True)


def _categories(manifest):
    return {e["category"] for e in manifest.get("errors", [])}


def _skipped_categories(manifest):
    return {s["category"] for s in manifest.get("skipped", [])}


def _fence_signature(text: str) -> list[str]:
    """Return the opening fence lines, in order, so we can assert byte-equality."""
    sigs: list[str] = []
    for m in _CODE_FENCE_RE.finditer(text):
        block = m.group(0)
        first_nl = block.find("\n")
        sigs.append(block[:first_nl])
        sigs.append("CLOSE")
    return sigs


# ── Detection helpers ──────────────────────────────────────────────────────


class TestLooksLikeCliLine:
    def test_launcher_or_script_token_required_even_in_bash(self):
        """A bash fence no longer makes every non-blank line eligible —
        the line still needs a launcher token or a script suffix so we
        don't mutate arbitrary command/output text."""
        inj = _make_injector()
        # A non-launcher non-script first token is rejected even in bash.
        assert inj._looks_like_cli_line("some_tool --foo bar", "bash") is False
        assert inj._looks_like_cli_line("EHOONAYF_CDS_0007", "bash") is False
        # Launcher and script forms still pass.
        assert inj._looks_like_cli_line("$ python train.py --epochs 5", "bash") is True
        assert inj._looks_like_cli_line("pharokka_plotter.py -i x.fasta", "bash") is True

    def test_python_launcher_in_non_shell_fence(self):
        inj = _make_injector()
        assert inj._looks_like_cli_line("python train.py --epochs 5", "") is True
        assert inj._looks_like_cli_line("python3 train.py", "python") is True

    def test_script_suffix_in_any_fence(self):
        inj = _make_injector()
        assert inj._looks_like_cli_line("pharokka_plotting.py -i x.tsv -o out/", "") is True
        assert inj._looks_like_cli_line("Rscript bin/analyze.R --input x.csv", "") is True

    def test_blank_or_comment_lines_rejected(self):
        inj = _make_injector()
        assert inj._looks_like_cli_line("", "bash") is False
        assert inj._looks_like_cli_line("# a comment", "bash") is False
        assert inj._looks_like_cli_line("    ", "bash") is False

    def test_plain_word_in_non_shell_fence_rejected(self):
        inj = _make_injector()
        # In a python fence, lines that aren't launcher/script don't qualify.
        assert inj._looks_like_cli_line("import os", "python") is False
        assert inj._looks_like_cli_line("library(Seurat)", "r") is False

    def test_argparse_help_columns_rejected(self):
        """Lines whose first token starts with `-` are argparse help rows."""
        inj = _make_injector()
        assert inj._looks_like_cli_line("  -i INFILE, --infile INFILE", "bash") is False
        assert inj._looks_like_cli_line("  --gff GFF             Pharokka gff.", "bash") is False
        assert inj._looks_like_cli_line("-h, --help show this help message", "bash") is False

    def test_argparse_stanza_headers_rejected(self):
        inj = _make_injector()
        assert inj._looks_like_cli_line("usage: pharokka.py [-h] -i INFILE", "bash") is False
        assert inj._looks_like_cli_line("options:", "bash") is False
        assert inj._looks_like_cli_line("positional arguments:", "bash") is False
        assert inj._looks_like_cli_line("optional arguments:", "bash") is False
        # Even with leading whitespace.
        assert inj._looks_like_cli_line("  usage: foo --bar", "bash") is False

    def test_usage_continuation_lines_rejected(self):
        """Continuations of the usage stanza start with bracketed tokens."""
        inj = _make_injector()
        assert inj._looks_like_cli_line(
            "                           [--label_hypotheticals] [--remove_other_features_labels]",
            "bash",
        ) is False

    def test_help_text_reference_rejected(self):
        """Lines that mention --help are explanatory, not invocations."""
        inj = _make_injector()
        assert inj._looks_like_cli_line(
            "  -h, --help show this help message and exit",
            "bash",
        ) is False
        assert inj._looks_like_cli_line(
            "python train.py --help  # for usage",
            "bash",
        ) is False


# ── Whitespace preservation ────────────────────────────────────────────────


PHAROKKA_USAGE_DOC = """\
# Pharokka help

```bash
usage: pharokka_plotter.py [-h] -i INFILE [-n PLOT_NAME] [-o OUTDIR] [--gff GFF]
                           [--label_hypotheticals] [--remove_other_features_labels]

options:
  -h, --help            show this help message and exit
  -i INFILE, --infile INFILE
                        Input genome file in FASTA format.
  --gff GFF             Pharokka gff.
```

Then run:

```bash
python scripts/pharokka_plotter.py --infile input.fasta --gff pharokka.gff
```
"""


class TestInlineCodeCandidates:
    """Pharokka-style docs put CLI invocations inside single-backtick inline
    code (e.g. `` `pharokka_plotter.py -i x.fasta --foo bar` ``) rather than
    in triple-backtick fenced blocks.  The injector must pick them up."""

    def test_inline_script_invocation_detected(self):
        text = (
            "Run like so:\n"
            "`pharokka_plotter.py -i input.fasta --label_ids labels.txt`\n"
        )
        inj = _make_injector()
        cands = inj._cli_inline_candidates(text)
        assert len(cands) == 1
        line, source = cands[0]
        assert line == "pharokka_plotter.py -i input.fasta --label_ids labels.txt"
        assert source == "inline"

    def test_inline_bare_flag_not_detected(self):
        """Prose mentions of a flag like `` `--gff` `` must NOT count — they
        aren't invocations, just option references."""
        text = "Use the `--gff` and `--genbank` options to specify files.\n"
        inj = _make_injector()
        assert inj._cli_inline_candidates(text) == []

    def test_inline_inside_fence_is_skipped(self):
        """Backticks that appear INSIDE a triple-backtick fence are part of
        the fenced content, not an inline-code span."""
        text = (
            "```bash\n"
            "echo `python train.py --epochs 5`\n"
            "```\n"
        )
        inj = _make_injector()
        # Inline scanner must skip this; the fence scanner picks up the line.
        inline = inj._cli_inline_candidates(text)
        assert inline == []

    def test_combined_candidates_include_both(self):
        text = (
            "```bash\n"
            "python scripts/train.py --epochs 5\n"
            "```\n\n"
            "Or inline: `pharokka_plotter.py -i input.fasta --gff x.gff`\n"
        )
        inj = _make_injector()
        cands = inj._cli_candidates(text)
        sources = [s for _, s in cands]
        assert sources == ["fence", "inline"], cands

    def test_inline_non_script_first_token_rejected(self):
        text = "Use `make build --verbose` to compile.\n"
        inj = _make_injector()
        assert inj._cli_inline_candidates(text) == []


class TestInlineCodeMutation:
    """When the candidate comes from inline code, the mutation must rewrite
    the inline span itself — backticks intact, surrounding prose unchanged."""

    def test_inline_cli_flag_typo_rewrites_inside_backticks(self):
        text = (
            "Then run:\n"
            "`pharokka_plotter.py -i input.fasta --label_ids labels.txt`\n"
            "to generate the plot.\n"
        )
        inj = _make_injector()
        result, errors = inj._inject_cli_consistency(text, [], {}, file_type=".md")
        typo = [e for e in errors if e["category"] == "cli_flag_typo"]
        assert typo, errors
        # Truncated flag (`--label_ids` → `--label_id`).
        assert typo[0]["mutated_token"] == "--label_id"
        # The mutated span still lives inside the inline-code backticks.
        assert "`pharokka_plotter.py -i input.fasta --label_id labels.txt`" in result
        # Surrounding prose untouched.
        assert "Then run:" in result and "to generate the plot." in result

    def test_inline_cli_unknown_flag_appends_inside_backticks(self):
        # Two distinct inline CLI lines so cli_flag_typo claims the first
        # and cli_unknown_flag still has the second to operate on.
        text = (
            "Plot: `pharokka_plotter.py -i input.fasta --label_ids labels.txt`\n"
            "Multi: `pharokka_multiplotter.py -g g.gbk -o out/ --label_size 12`\n"
        )
        inj = _make_injector()
        result, errors = inj._inject_cli_consistency(text, [], {}, file_type=".md")
        bogus = [e for e in errors if e["category"] == "cli_unknown_flag"]
        assert bogus, errors
        token = bogus[0]["mutated_token"]
        # The bogus flag is now inside an inline-code span (still backticked).
        assert f"{token} " in bogus[0]["mutated_snippet"]
        assert f"`{bogus[0]['mutated_snippet']}`" in result
        # The token should not appear in prose outside of inline code.
        # Strip every inline-code span and assert no bogus token leaks out.
        import re as _re
        prose_only = _re.sub(r"`[^`\n]+`", "", result)
        assert token not in prose_only

    def test_inline_pharokka_style_doc_yields_real_candidates(self):
        """The pharokka-style mini-doc — many inline `script.py --flag …`
        invocations, no fenced block — should yield CLI candidates so the
        injector has something to mutate."""
        text = (
            "`pharokka_plotter.py -i input.fasta --label_hypotheticals`\n\n"
            "`pharokka_plotter.py -i input.fasta -n p -o out --truncate 15`\n\n"
            "`pharokka_plotter.py -i input.fasta -n p --gff x.gff --genbank y.gbk`\n"
        )
        inj = _make_injector()
        cands = inj._cli_inline_candidates(text)
        assert len(cands) == 3
        # All inline.
        assert all(src == "inline" for _, src in cands)


class TestPharokkaHelpBlockUntouched:
    def test_help_block_lines_are_not_mutated(self):
        """The whole argparse help stanza should be left alone; only the real
        invocation in the second fence should be mutated."""
        inj = _make_injector()
        errors: list = []
        data: dict = {}
        result, errors = inj._inject_cli_consistency(
            PHAROKKA_USAGE_DOC, errors, data, file_type=".md"
        )

        # Every help-block line from the input must still appear verbatim.
        help_lines = [
            "usage: pharokka_plotter.py [-h] -i INFILE [-n PLOT_NAME] [-o OUTDIR] [--gff GFF]",
            "                           [--label_hypotheticals] [--remove_other_features_labels]",
            "options:",
            "  -h, --help            show this help message and exit",
            "  -i INFILE, --infile INFILE",
            "                        Input genome file in FASTA format.",
            "  --gff GFF             Pharokka gff.",
        ]
        for hl in help_lines:
            assert hl in result, f"help-block line was mutated: {hl!r}"

        # Errors should target only the real-invocation line.
        for e in errors:
            assert "python scripts/pharokka_plotter.py" in e["original_snippet"], (
                f"injection targeted help-block line: {e!r}"
            )


class TestMutatedTokenRecorded:
    """The manifest must carry the precise token introduced/changed by each
    CLI mutation, so the evaluator can do a token-anchored fix check."""

    def test_cli_flag_typo_records_truncated_flag(self):
        text = (
            "```bash\n"
            "python scripts/train.py --epochs 20\n"
            "```\n"
        )
        inj = _make_injector()
        _result, errors = inj._inject_cli_consistency(text, [], {}, file_type=".md")
        typo = [e for e in errors if e["category"] == "cli_flag_typo"]
        assert typo
        # Truncated form of --epochs, with no surrounding whitespace.
        assert typo[0]["mutated_token"] == "--epoch"

    def test_cli_unknown_flag_records_bogus_flag(self):
        text = (
            "```bash\n"
            "python scripts/train.py --epochs 20\n"
            "```\n\n"
            "```bash\n"
            "python scripts/evaluate.py --metric accuracy\n"
            "```\n"
        )
        inj = _make_injector()
        _result, errors = inj._inject_cli_consistency(text, [], {}, file_type=".md")
        bogus = [e for e in errors if e["category"] == "cli_unknown_flag"]
        assert bogus
        bogus_names = {f for f, _ in _CLI_BOGUS_FLAGS}
        # Token is the flag *name only* (no value), drawn from the round-robin set.
        assert bogus[0]["mutated_token"] in bogus_names

    def test_cli_program_rename_records_mutated_program(self):
        text = (
            "```bash\n"
            "python scripts/train.py --epochs 20\n"
            "```\n\n"
            "```bash\n"
            "python scripts/evaluate.py --metric accuracy\n"
            "```\n\n"
            "```bash\n"
            "python scripts/serve.py --port 8080\n"
            "```\n"
        )
        inj = _make_injector()
        _result, errors = inj._inject_cli_consistency(text, [], {}, file_type=".md")
        prog = [e for e in errors if e["category"] == "cli_program_rename"]
        assert prog
        token = prog[0]["mutated_token"]
        # Token is the mutated script basename, still ending in .py.
        assert token.endswith(".py")
        assert token != "train.py" and token != "evaluate.py" and token != "serve.py"


class TestWhitespacePreservedInMutation:
    def test_cli_flag_typo_preserves_surrounding_alignment(self):
        """Multi-space gaps elsewhere on the line stay intact when one token swaps."""
        text = (
            "```bash\n"
            "python scripts/train.py   --epochs 20   --lr 0.001\n"
            "```\n"
        )
        inj = _make_injector()
        result, errors = inj._inject_cli_consistency(text, [], {}, file_type=".md")
        typo = [e for e in errors if e["category"] == "cli_flag_typo"]
        assert typo, errors
        mut_line = typo[0]["mutated_snippet"]
        # The "   --epochs " and "   --lr 0.001" alignment columns stay.
        assert "scripts/train.py   " in mut_line
        assert "   --lr 0.001" in mut_line
        # One char dropped from --epochs.
        assert "--epoch " in mut_line and "--epochs " not in mut_line

    def test_cli_program_rename_preserves_surrounding_alignment(self):
        # Three CLI fences — cli_flag_typo and cli_unknown_flag will each
        # claim one, leaving the third for cli_program_rename.
        text = (
            "```bash\n"
            "python scripts/train.py --epochs 20\n"
            "```\n\n"
            "```bash\n"
            "python scripts/evaluate.py    --metric    accuracy\n"
            "```\n\n"
            "```bash\n"
            "python scripts/serve.py --port 8080\n"
            "```\n"
        )
        inj = _make_injector()
        _result, errors = inj._inject_cli_consistency(text, [], {}, file_type=".md")
        prog = [e for e in errors if e["category"] == "cli_program_rename"]
        assert prog, errors
        orig_line = prog[0]["original_snippet"]
        mut_line = prog[0]["mutated_snippet"]
        # Whitespace runs (the odd indices of re.split with a captured group)
        # must match byte-for-byte between original and mutation — only one
        # non-whitespace token has changed.
        import re as _re
        orig_ws = _re.split(r"(\s+)", orig_line)[1::2]
        mut_ws = _re.split(r"(\s+)", mut_line)[1::2]
        assert orig_ws == mut_ws, (orig_line, mut_line)


class TestPickHelpers:
    def test_pick_long_flag_skips_short_and_protected(self):
        inj = _make_injector()
        assert inj._pick_long_flag(["python", "train.py", "-i", "x"]) is None
        assert inj._pick_long_flag(["python", "train.py", "--help"]) is None
        # First eligible is --epochs.
        idx, t = inj._pick_long_flag(["python", "train.py", "--help", "--epochs", "20"])
        assert (idx, t) == (3, "--epochs")

    def test_pick_long_flag_handles_equals_form(self):
        inj = _make_injector()
        idx, t = inj._pick_long_flag(["python", "train.py", "--epochs=20"])
        assert t == "--epochs=20"

    def test_truncate_long_flag_with_and_without_equals(self):
        inj = _make_injector()
        assert inj._truncate_long_flag("--epochs") == "--epoch"
        assert inj._truncate_long_flag("--epochs=20") == "--epoch=20"
        # Too short to safely truncate.
        assert inj._truncate_long_flag("--ab") == "--ab"

    def test_pick_program_token_recognises_script(self):
        inj = _make_injector()
        assert inj._pick_program_token(["python", "scripts/train.py", "--epochs", "5"]) == (
            1, "scripts/train.py"
        )
        assert inj._pick_program_token(["pharokka_plotting.py", "-i", "x"]) == (
            0, "pharokka_plotting.py"
        )

    def test_pick_program_token_rejects_non_script(self):
        inj = _make_injector()
        assert inj._pick_program_token(["make", "build"]) is None
        assert inj._pick_program_token(["python", "-m", "mymodule"]) is None
        assert inj._pick_program_token([]) is None

    def test_mutate_program_token_preserves_path_and_extension(self):
        inj = _make_injector()
        mutated = inj._mutate_program_token("scripts/pharokka_plotting.py")
        assert mutated != "scripts/pharokka_plotting.py"
        assert mutated.startswith("scripts/")
        assert mutated.endswith(".py")


# ── _inject_cli_consistency — direct tests ────────────────────────────────


CLI_DOC_BASH = """\
# Tool

Train the model:

```bash
python scripts/train.py --epochs 20 --lr 0.001 data/train.h5
```
"""


CLI_DOC_PY_FENCE = """\
# Tool

```python
python scripts/train.py --epochs 20
```
"""


CLI_DOC_RSCRIPT = """\
# R Tool

```bash
Rscript bin/analyze.R --input data.csv --threshold 0.05
```
"""


CLI_DOC_MULTI = """\
# Tool

First train:

```bash
python scripts/train.py --epochs 20
```

Then evaluate:

```bash
python scripts/evaluate.py --metric accuracy
```

Finally serve:

```bash
python scripts/serve.py --port 8080
```
"""


class TestInjectCliConsistencyDirect:
    def _call(self, text, file_type=".md"):
        inj = _make_injector()
        errors: list = []
        data: dict = {}
        result, errors = inj._inject_cli_consistency(text, errors, data, file_type=file_type)
        return result, errors, data

    def test_cli_flag_typo_truncates_long_flag(self):
        result, errors, _ = self._call(CLI_DOC_BASH)
        typo_errors = [e for e in errors if e["category"] == "cli_flag_typo"]
        assert typo_errors, errors
        e = typo_errors[0]
        # The original line was the train command; the mutated line should have
        # a truncated long flag (the first eligible long flag is --epochs).
        assert "--epochs" in e["original_snippet"]
        assert "--epoch " in e["mutated_snippet"] or "--epoch" in e["mutated_snippet"]
        # The mutated line must appear in the resulting text.
        assert e["mutated_snippet"] in result

    def test_cli_unknown_flag_appends_bogus_flag(self):
        result, errors, _ = self._call(CLI_DOC_BASH)
        bogus_errors = [e for e in errors if e["category"] == "cli_unknown_flag"]
        if not bogus_errors:
            pytest.skip("only one CLI line; cli_unknown_flag couldn't pick a fresh line")
        e = bogus_errors[0]
        bogus_names = {f for f, _ in _CLI_BOGUS_FLAGS}
        assert any(name in e["mutated_snippet"] for name in bogus_names)

    def test_cli_program_rename_mutates_script_token(self):
        result, errors, _ = self._call(CLI_DOC_MULTI)
        prog_errors = [e for e in errors if e["category"] == "cli_program_rename"]
        assert prog_errors, errors
        e = prog_errors[0]
        # Original carried a real .py token; mutated carries a different .py
        # token (transposed stem) on the same line.
        assert ".py" in e["original_snippet"] and ".py" in e["mutated_snippet"]
        assert e["original_snippet"] != e["mutated_snippet"]

    def test_three_categories_pick_distinct_lines_when_multiple_available(self):
        result, errors, _ = self._call(CLI_DOC_MULTI)
        cats = [e["category"] for e in errors]
        # All three should fire with three available CLI lines.
        assert {"cli_flag_typo", "cli_unknown_flag", "cli_program_rename"}.issubset(set(cats))

    def test_fence_delimiters_preserved(self):
        result, _errors, _ = self._call(CLI_DOC_MULTI)
        # Same number and order of fence opens; closes likewise present.
        assert _fence_signature(result) == _fence_signature(CLI_DOC_MULTI)
        # Triple-backtick counts identical.
        assert result.count("```") == CLI_DOC_MULTI.count("```")

    def test_rscript_line_in_bash_fence_is_eligible_for_typo(self):
        result, errors, _ = self._call(CLI_DOC_RSCRIPT)
        typo = [e for e in errors if e["category"] == "cli_flag_typo"]
        assert typo, errors
        assert "--input" in typo[0]["original_snippet"]

    def test_rmd_file_type_records_skips(self):
        _result, errors, data = self._call(CLI_DOC_BASH, file_type=".rmd")
        assert errors == []
        skipped = _skipped_categories(data)
        assert {"cli_flag_typo", "cli_unknown_flag", "cli_program_rename"} <= skipped
        for s in data["skipped"]:
            assert s["reason"] == "executable_code_file"

    def test_unknown_file_type_records_skips(self):
        _result, errors, data = self._call(CLI_DOC_BASH, file_type=".yaml")
        assert errors == []
        skipped = _skipped_categories(data)
        assert {"cli_flag_typo", "cli_unknown_flag", "cli_program_rename"} <= skipped

    def test_no_code_block_records_skips(self):
        _result, errors, data = self._call("# Just prose, no fences.\n")
        assert errors == []
        skipped = _skipped_categories(data)
        assert {"cli_flag_typo", "cli_unknown_flag", "cli_program_rename"} <= skipped

    def test_fence_without_cli_lines_records_skips(self):
        text = "```python\nimport os\nx = 1\n```\n"
        _result, errors, data = self._call(text)
        assert errors == []
        skipped = _skipped_categories(data)
        assert {"cli_flag_typo", "cli_unknown_flag", "cli_program_rename"} <= skipped


# ── End-to-end via _deterministic_inject ──────────────────────────────────


class TestDeterministicInjectWithCli:
    def test_cli_categories_appear_in_full_deterministic_path(self):
        inj = _make_injector()
        corrupted, data = inj.inject(CLI_DOC_MULTI, file_type=".md")
        cats = _categories(data)
        # At least one CLI category fires when CLI lines are available.
        assert cats & {"cli_flag_typo", "cli_unknown_flag", "cli_program_rename"}, data
        # Fences still intact byte-for-byte.
        assert corrupted.count("```") == CLI_DOC_MULTI.count("```")
