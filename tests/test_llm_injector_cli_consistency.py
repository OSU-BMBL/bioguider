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
        # Permuted flag (`--label_ids` → some transposition of the stem).
        mut_token = typo[0]["mutated_token"]
        assert mut_token.startswith("--") and mut_token != "--label_ids"
        assert sorted(mut_token) == sorted("--label_ids")
        # The mutated span still lives inside the inline-code backticks.
        assert f"`pharokka_plotter.py -i input.fasta {mut_token} labels.txt`" in result
        # Original flag must be gone everywhere.
        assert "--label_ids" not in result
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
    def test_help_block_is_never_a_picking_target(self):
        """The argparse help stanza must never be the *source* of a picked
        flag/program/bogus-flag injection — picks come from real invocation
        lines only.  (The mutated flag may still ripple into help-block
        lines via global rewriting; see
        ``test_picked_flag_propagates_into_help_block`` for that side.)"""
        inj = _make_injector()
        errors: list = []
        data: dict = {}
        _result, errors = inj._inject_cli_consistency(
            PHAROKKA_USAGE_DOC, errors, data, file_type=".md"
        )
        for e in errors:
            assert "python scripts/pharokka_plotter.py" in e["original_snippet"], (
                f"injection targeted help-block line: {e!r}"
            )

    def test_picked_flag_propagates_into_help_block(self):
        """cli_flag_typo is now document-wide: when the injector picks
        ``--infile`` (or any other flag) from the invocation, every other
        mention of that flag — including the help stanza — gets rewritten so
        the doc is internally consistent in its wrongness."""
        inj = _make_injector()
        errors: list = []
        data: dict = {}
        result, errors = inj._inject_cli_consistency(
            PHAROKKA_USAGE_DOC, errors, data, file_type=".md"
        )
        typo = [e for e in errors if e["category"] == "cli_flag_typo"]
        assert typo, errors
        e = typo[0]
        mut_flag = e["mutated_token"]
        orig_flag = e["original_token"]
        # Picked flag's original form must be wholly gone from the document.
        import re as _re
        leftover = _re.search(rf"(?<![\w-]){_re.escape(orig_flag)}(?![\w-])", result)
        assert leftover is None, (
            f"global rewrite missed an occurrence of {orig_flag!r}: ...{result[max(0,leftover.start()-20):leftover.end()+20]}..."
            if leftover else ""
        )
        # The mutated flag should appear in MORE than one place if the
        # original appeared in both the invocation and the help block.
        # ``--infile`` appears twice in PHAROKKA_USAGE_DOC (help + invoke);
        # other picks (``--gff``) likewise appear in both.
        assert result.count(mut_flag) >= 2, (
            f"expected >=2 occurrences of {mut_flag} after global rewrite, "
            f"got {result.count(mut_flag)}"
        )
        # occurrences_changed should reflect the same count.
        assert e.get("occurrences_changed", 0) >= 2


class TestMutatedTokenRecorded:
    """The manifest must carry the precise token introduced/changed by each
    CLI mutation, so the evaluator can do a token-anchored fix check."""

    def test_cli_flag_typo_records_permuted_flag(self):
        text = (
            "```bash\n"
            "python scripts/train.py --epochs 20\n"
            "```\n"
        )
        inj = _make_injector()
        _result, errors = inj._inject_cli_consistency(text, [], {}, file_type=".md")
        typo = [e for e in errors if e["category"] == "cli_flag_typo"]
        assert typo
        # Permuted form of --epochs (two adjacent stem chars transposed),
        # same length as the original — no characters dropped.
        token = typo[0]["mutated_token"]
        assert token.startswith("--") and len(token) == len("--epochs")
        assert token != "--epochs"

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
        mut_token = typo[0]["mutated_token"]
        # The "   --epochs " and "   --lr 0.001" alignment columns stay —
        # permutation preserves token length so columns line up byte-for-byte.
        assert "scripts/train.py   " in mut_line
        assert "   --lr 0.001" in mut_line
        # The flag was permuted (same length, different ordering).
        assert mut_token in mut_line and "--epochs" not in mut_line
        assert len(mut_token) == len("--epochs")

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

    def test_permute_long_flag_with_and_without_equals(self):
        inj = _make_injector()
        # Permuted form: same length, leading "--" intact, stem reordered.
        out = inj._permute_long_flag("--epochs")
        assert out.startswith("--") and out != "--epochs"
        assert len(out) == len("--epochs")
        assert sorted(out) == sorted("--epochs")
        # ``=value`` suffix is preserved unchanged.
        out_eq = inj._permute_long_flag("--epochs=20")
        assert out_eq.endswith("=20")
        assert out_eq.split("=", 1)[0] == out
        # Too short to safely transpose — returns the original.
        assert inj._permute_long_flag("--ab") == "--ab"

    def test_replace_flag_globally_hits_all_standalone_occurrences(self):
        inj = _make_injector()
        text = (
            "Use --foo on its own; in `--foo bar` inline; in `[--foo]` brackets; "
            "and --foo=value with an equals."
        )
        out, n = inj._replace_flag_globally(text, "--foo", "--ofo")
        assert n == 4, out
        assert "--foo" not in out
        assert out.count("--ofo") == 4

    def test_replace_flag_globally_does_not_overshoot(self):
        inj = _make_injector()
        text = "use --foo here, but not --foo-bar, --foobar, or --foo_x"
        out, n = inj._replace_flag_globally(text, "--foo", "--ofo")
        # Only the standalone --foo is replaced.
        assert n == 1
        assert "--foo-bar" in out and "--foobar" in out and "--foo_x" in out

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

    def test_cli_flag_typo_permutes_long_flag(self):
        result, errors, _ = self._call(CLI_DOC_BASH)
        typo_errors = [e for e in errors if e["category"] == "cli_flag_typo"]
        assert typo_errors, errors
        e = typo_errors[0]
        # The original line was the train command; the picked flag is the
        # first eligible long flag (--epochs).
        assert "--epochs" in e["original_snippet"]
        mut_token = e["mutated_token"]
        assert mut_token.startswith("--") and mut_token != "--epochs"
        assert sorted(mut_token) == sorted("--epochs")  # same chars, reordered
        # The mutated line must appear in the resulting text.
        assert e["mutated_snippet"] in result
        # Original flag must no longer appear anywhere (global rewrite).
        assert "--epochs" not in result

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

    def test_cli_flag_typo_rewrites_every_occurrence_globally(self):
        """When the picked flag appears in multiple places — fenced block,
        inline-code, prose mention — every occurrence must be rewritten so
        the document is internally consistent in its wrongness."""
        text = (
            "# Tool\n\n"
            "Pass `--annotations` to enable annotations.\n\n"
            "```bash\n"
            "python scripts/train.py --annotations metadata.tsv\n"
            "```\n\n"
            "You can also override via `train.py --annotations=other.tsv`.\n\n"
            "See [the `--annotations` option] for details.\n"
        )
        result, errors, _ = self._call(text)
        typo = [e for e in errors if e["category"] == "cli_flag_typo"]
        assert typo, errors
        e = typo[0]
        assert e["original_token"] == "--annotations"
        mut_flag = e["mutated_token"]
        # All four original occurrences (prose mention, fenced invocation,
        # ``=value`` form, bracketed mention) must be rewritten.
        assert "--annotations" not in result
        assert result.count(mut_flag) == 4
        assert e["occurrences_changed"] == 4

    def test_cli_flag_typo_does_not_overshoot_longer_flag_names(self):
        """Replacing ``--label`` must not also rewrite ``--label-extra`` or
        ``--label_size``: the token boundary must be flag-name aware."""
        text = (
            "```bash\n"
            "python scripts/train.py --label data.tsv\n"
            "```\n\n"
            "Related: `--label-extra` and `--label_size` are different flags.\n"
        )
        # --label is only 7 chars (>4 so eligible), but the test bites only
        # if --label is what the picker chose.  Run and inspect.
        result, errors, _ = self._call(text)
        typo = [e for e in errors if e["category"] == "cli_flag_typo"]
        assert typo, errors
        if typo[0]["original_token"] != "--label":
            pytest.skip("picker chose a different flag in this fixture")
        # The longer-named siblings must be untouched.
        assert "--label-extra" in result
        assert "--label_size" in result
        # ``--label`` itself must be globally rewritten.
        assert "--label " not in result and "--label\n" not in result

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


class TestCliFlagTypoManifestMatchesFile:
    """Every cli_flag_typo manifest entry must correspond to a mutation that
    actually persists in the corrupted document.  The earlier supplements loop
    silently ping-pong'd back its own mutations (picking the post-mutated form
    on the next iteration and ``_permute_long_flag``-ing it BACK to the
    original) so the manifest counted phantom fixes.  This regression test
    guards against that: for every recorded entry, the original flag must be
    gone and the mutated flag must be present at the recorded occurrence count.
    """

    def _doc_with_many_flags(self) -> str:
        # Multiple distinct long flags, each appearing several times across
        # fenced blocks and inline-code spans, plus one CLI line per fence so
        # the picker has enough material for many supplement iterations.
        # NOTE: avoid ``--flag=value`` forms in prose — the ``param_name``
        # supplement matches ``identifier=`` patterns and will happily mutate
        # ``--annottaions=meta.tsv`` to ``--annottaion=meta.tsv``, which is a
        # separate interaction from the cli_flag_typo behaviour we want to pin.
        return (
            "# Tool\n\n"
            "Pass `--annotations` to enable annotation parsing.\n\n"
            "```bash\n"
            "python scripts/train.py --annotations data/meta.tsv\n"
            "python scripts/train.py --annotations other.tsv\n"
            "```\n\n"
            "Also: `--label_hypotheticals` for label control.\n\n"
            "```bash\n"
            "python scripts/run.py --label_hypotheticals --truncate 15\n"
            "```\n\n"
            "Use `--truncate 30` for shorter output.\n\n"
            "```bash\n"
            "python scripts/analyze.py --label_size 24 --label_ids ids.txt\n"
            "```\n"
        )

    def test_every_cli_flag_typo_mutation_lands_in_text(self):
        inj = _make_injector()
        text = self._doc_with_many_flags()
        corrupted, manifest = inj.inject(text, min_per_category=10, file_type=".md")
        typo_errors = [e for e in manifest["errors"] if e["category"] == "cli_flag_typo"]
        assert typo_errors, manifest
        import re as _re
        for e in typo_errors:
            ot = e["original_token"]
            mt = e["mutated_token"]
            n = e["occurrences_changed"]
            ot_pat = _re.compile(rf"(?<![\w-]){_re.escape(ot)}(?![\w-])")
            mt_pat = _re.compile(rf"(?<![\w-]){_re.escape(mt)}(?![\w-])")
            assert len(ot_pat.findall(corrupted)) == 0, (
                f"manifest says {ot!r} was rewritten {n} times, but it is still "
                f"present in corrupted text — phantom mutation"
            )
            assert len(mt_pat.findall(corrupted)) >= n, (
                f"manifest says {mt!r} should appear at least {n} times after "
                f"global rewrite, but it appears {len(mt_pat.findall(corrupted))} times"
            )

    def test_supplement_loop_does_not_pick_bogus_flags(self):
        """``--workers``/``--nproc``/``--threads``/``--cores`` are appended by
        cli_unknown_flag; cli_flag_typo must not then permute one of them
        (which would create entries that are unstable across iterations)."""
        from bioguider.generation.llm_injector import _CLI_BOGUS_FLAGS
        inj = _make_injector()
        text = self._doc_with_many_flags()
        _corrupted, manifest = inj.inject(text, min_per_category=10, file_type=".md")
        bogus_names = {f for f, _ in _CLI_BOGUS_FLAGS}
        for e in manifest["errors"]:
            if e["category"] == "cli_flag_typo":
                ot = e.get("original_token", "")
                mt = e.get("mutated_token", "")
                assert ot not in bogus_names, f"picked a bogus flag as orig: {e!r}"
                assert mt not in bogus_names, f"mutated to a bogus flag: {e!r}"


# ===========================================================================
# Help-block consistency for cli_unknown_flag
# ===========================================================================
#
# Closes the loophole the user spotted in pharokka: the corrupted doc
# preserved the canonical ``usage:`` / ``options:`` stanza, and a model
# with no repo access could trivially cross-reference it to strip the
# appended bogus flag.  These tests pin that the splicer:
#
#   * adds the bogus flag to BOTH the usage line and the options entry,
#   * stays idempotent on repeat calls (the supplement loop runs ≥1 time),
#   * no-ops cleanly on docs without a help block (so non-CLI-tooling
#     repos still get the legacy single-site injection without crashing),
#   * is wired into both the primary and supplement injection paths.

PHAROKKA_HELP_DOC = """\
# pharokka_plotter

```text
usage: pharokka_plotter.py [-h] -i INFILE [-o OUTDIR]
                           [--label_hypotheticals] [--annotations ANNOTATIONS]

pharokka_plotter.py: pharokka plotting function

options:
  -h, --help            show this help message and exit
  -i INFILE, --infile INFILE
                        Input genome file.
  --label_hypotheticals
                        Flag to label hypothetical proteins.
  --annotations ANNOTATIONS
                        Annotation density.
```

Run it:
```bash
pharokka_plotter.py -i input.fasta -o out --label_hypotheticals
```

Or with annotations:
```bash
pharokka_plotter.py -i input.fasta -o out --annotations 0.5
```

For multiple plots:
```bash
pharokka_plotter.py -i input.fasta -n p1 -o out -t 'phage' --label_hypotheticals
```

And a prefix:
```bash
pharokka_plotter.py -i input.fasta -n p2 -o out -p myprefix --annotations 1.0
```

Larger plot title:
```bash
pharokka_plotter.py -i input.fasta -n p3 -o out -t 'big phage' --label_hypotheticals --annotations 0.8
```
"""


class TestSpliceBogusFlagIntoHelpBlock:
    """Direct unit tests on the splicer (static method, no LLM)."""

    def test_splices_into_usage_and_options(self):
        new, n = LLMErrorInjector._splice_bogus_flag_into_help_block(
            PHAROKKA_HELP_DOC, "--workers", "4"
        )
        assert n == 2  # one usage edit + one options edit
        # Usage line picks up ``[--workers VAL]`` (numeric value → VAL token).
        assert "[--workers VAL]" in new
        # Options section gets a new entry, matching argparse's 2-space
        # indent + 24-space description indent.
        assert "  --workers VAL\n                        Number of parallel workers." in new

    def test_idempotent_on_second_call_same_flag(self):
        """The supplement loop runs the splicer repeatedly with the same
        bogus flag during round-robin reuse — a second call must NOT
        double-insert into either surface."""
        once, _ = LLMErrorInjector._splice_bogus_flag_into_help_block(
            PHAROKKA_HELP_DOC, "--workers", "4"
        )
        twice, n2 = LLMErrorInjector._splice_bogus_flag_into_help_block(
            once, "--workers", "4"
        )
        assert n2 == 0
        assert twice == once
        # Sanity: each surface still contains exactly one occurrence.
        assert once.count("[--workers VAL]") == 1
        assert once.count("  --workers VAL\n") == 1

    def test_no_op_when_no_help_block(self):
        plain = "## Plot\n\nRun: `pharokka_plotter.py -i in.fa`\n"
        out, n = LLMErrorInjector._splice_bogus_flag_into_help_block(plain, "--workers", "4")
        assert n == 0
        assert out == plain

    def test_no_op_when_flag_already_present_in_usage(self):
        """Edge case: the doc claims ``--workers`` is valid (lists it in
        usage).  Don't double-add; the splicer no-ops on that surface."""
        doc = (
            "```text\n"
            "usage: prog.py [-h] [--workers WORKERS]\n"
            "\n"
            "options:\n"
            "  -h, --help  show help\n"
            "```\n"
        )
        out, n = LLMErrorInjector._splice_bogus_flag_into_help_block(doc, "--workers", "4")
        # Usage already has it → 0 edits there; options body lacks an entry
        # → 1 edit there.  Tests the per-surface decision, not all-or-nothing.
        assert n == 1
        assert out.count("[--workers WORKERS]") == 1
        assert "  --workers VAL\n" in out

    def test_handles_optional_arguments_header(self):
        """Older argparse (Python 3.9 and below) used ``optional arguments:``
        instead of ``options:`` — the regex covers both."""
        doc = (
            "```\n"
            "usage: prog.py [-h]\n"
            "\n"
            "optional arguments:\n"
            "  -h, --help  show help\n"
            "```\n"
        )
        out, n = LLMErrorInjector._splice_bogus_flag_into_help_block(doc, "--threads", "2")
        assert n == 2
        assert "[--threads VAL]" in out
        assert "  --threads VAL\n" in out

    def test_does_not_match_token_inside_longer_flag_name(self):
        """Presence check must be whitespace-anchored — ``--workers`` should
        not be considered "already present" inside ``--workers-pool``."""
        doc = (
            "```\n"
            "usage: prog.py [-h] [--workers-pool POOL]\n"
            "\n"
            "options:\n"
            "  --workers-pool POOL  configure the pool\n"
            "```\n"
        )
        out, n = LLMErrorInjector._splice_bogus_flag_into_help_block(doc, "--workers", "4")
        assert n == 2, "splicer mis-detected --workers as already present"
        assert "[--workers VAL]" in out
        assert "  --workers VAL\n" in out


class TestCliUnknownFlagWithHelpBlockInjection:
    """End-to-end through ``inject(...)``: confirm the manifest records the
    help-block edit AND the corrupted doc carries the bogus flag in the
    help block as well as in a shell invocation."""

    def test_primary_path_splices_into_help_block(self):
        inj = _make_injector()
        # min_per_category=1 keeps it to the primary path only.
        corrupted, manifest = inj.inject(
            PHAROKKA_HELP_DOC, min_per_category=1, file_type=".md"
        )
        cuf = [e for e in manifest["errors"] if e["category"] == "cli_unknown_flag"]
        assert cuf, "no cli_unknown_flag entry was produced"
        entry = cuf[0]
        # Manifest must record the help-block edit count.
        assert entry.get("help_block_edits") == 2, entry
        # The bogus flag now lives in the help block in addition to the
        # shell line — this is what defeats help-block grounding.
        token = entry["mutated_token"]
        assert f"[{token} VAL]" in corrupted, "usage line was not edited"
        assert f"  {token} VAL\n" in corrupted, "options entry was not added"

    def test_supplement_path_splices_and_stays_idempotent(self):
        """High min_per_category triggers the supplement loop multiple
        times.  Same bogus flag may be re-picked via the round-robin —
        each repeat must be a no-op on the help block surface."""
        inj = _make_injector()
        corrupted, manifest = inj.inject(
            PHAROKKA_HELP_DOC, min_per_category=6, file_type=".md"
        )
        cuf = [e for e in manifest["errors"] if e["category"] == "cli_unknown_flag"]
        assert len(cuf) >= 2, "expected supplement loop to produce ≥2 entries"
        # Each DISTINCT bogus flag injected must show up exactly once in
        # the usage line and exactly once in the options body, regardless
        # of how many cli_unknown_flag entries used it.
        distinct_tokens = {e["mutated_token"] for e in cuf}
        for tok in distinct_tokens:
            assert corrupted.count(f"[{tok} VAL]") == 1, f"duplicate usage entry for {tok}"
            assert corrupted.count(f"  {tok} VAL\n") == 1, f"duplicate options entry for {tok}"
        # The FIRST entry (primary path) records 2 edits.  Later entries
        # for the same flag should record 0; entries for fresh flags
        # should record 2.  At least one entry must record 2 (the bug we
        # are fixing) — flag this regression if it ever silently flips to
        # 0 everywhere.
        edits = [e.get("help_block_edits", 0) for e in cuf]
        assert max(edits) == 2, f"no entry actually edited the help block: {edits}"

    def test_doc_without_help_block_still_injects_legacy_way(self):
        """Repos without an embedded ``--help`` stanza must still get the
        single-site cli_unknown_flag injection — this is the original
        behaviour and must NOT regress for docs the splicer can't act on.

        Two distinct CLI lines so cli_flag_typo claims one and
        cli_unknown_flag still has something to operate on (cli_flag_typo
        is processed first within the same fence and consumes a line)."""
        doc = (
            "# pharokka\n\nRun:\n\n"
            "```bash\n"
            "pharokka_plotter.py -i input.fasta -o out --label_size 12\n"
            "```\n\n"
            "Or with truncation:\n\n"
            "```bash\n"
            "pharokka_plotter.py -i input.fasta -o out --truncate 25\n"
            "```\n"
        )
        inj = _make_injector()
        corrupted, manifest = inj.inject(doc, min_per_category=1, file_type=".md")
        cuf = [e for e in manifest["errors"] if e["category"] == "cli_unknown_flag"]
        assert cuf, "legacy single-site injection should still produce an entry"
        # No help block → no help-block edits, but mutated_token and the
        # shell-line append still land normally.
        assert cuf[0]["help_block_edits"] == 0
        assert cuf[0]["mutated_token"] in corrupted
