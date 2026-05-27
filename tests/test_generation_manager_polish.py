"""Integration tests for the markdown-polish step inside
``DocumentationGenerationManager``.

The manager runs a 9-step pipeline that ends with per-file rendering.
Polish was slotted in at the very end of the per-file loop, immediately
BEFORE ``LLMCleaner``:

  ... → _process_file_edits → _polish_content → _clean_content → revised[fpath]

These tests pin three things that any future refactor must preserve:

  * ``GenerationConfig.polish_output`` defaults to True — flipping it
    silently regresses the simple-vs-pipeline gap on every existing call
    site that doesn't pass an explicit config.
  * ``_polish_content`` runs on the same LLM as the rest of the manager;
    a hidden default-model factory would silently swap models the way
    ``_generate_continuation`` used to (now fixed and pinned elsewhere).
  * Polish runs BEFORE the cleaner.  Reversing the order would let
    cleaner-introduced AI-summary tells slip past polish, and would let
    polish-introduced tells slip past cleaner — both regressions.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from bioguider.managers.config import GenerationConfig
from bioguider.managers.generation_manager import DocumentationGenerationManager


# ---------------------------------------------------------------------------
# Default-value pins
# ---------------------------------------------------------------------------

class TestGenerationConfigPolishDefault:
    def test_polish_output_defaults_to_true(self):
        """The default is load-bearing — every existing benchmark / production
        run that constructs ``GenerationConfig()`` without overrides expects
        polish to be ON.  A False default would silently disable the win."""
        assert GenerationConfig().polish_output is True

    def test_polish_can_be_disabled_for_ablation(self):
        cfg = GenerationConfig(polish_output=False)
        assert cfg.polish_output is False


# ---------------------------------------------------------------------------
# Construction — polisher is wired to the manager's LLM
# ---------------------------------------------------------------------------

class TestManagerPolisherConstruction:
    def test_polisher_constructed_with_manager_llm(self):
        """Same invariant we pinned for ``_generate_continuation``: the
        polish call must be attributed to the model under test, never to
        a hard-coded default."""
        llm = MagicMock(name="model-under-test")
        mgr = DocumentationGenerationManager(
            llm=llm, step_callback=None, output_dir="/tmp/out"
        )
        assert mgr.polisher.llm is llm


# ---------------------------------------------------------------------------
# _polish_content — skip rules + happy path + failure handling
# ---------------------------------------------------------------------------

def _make_manager(*, polish_output: bool = True, clean_output: bool = True):
    """Build a manager with a mock LLM and the polish/clean flags wired."""
    llm = MagicMock()
    cfg = GenerationConfig(
        output_dir="/tmp/out",
        polish_output=polish_output,
        clean_output=clean_output,
    )
    return DocumentationGenerationManager(llm=llm, step_callback=None, config=cfg)


class TestPolishContentSkipRules:
    def test_returns_empty_when_content_empty(self):
        mgr = _make_manager()
        mgr.polisher = MagicMock()
        assert mgr._polish_content("foo.md", "") == ""
        mgr.polisher.polish.assert_not_called()

    def test_skips_non_markdown_extensions(self):
        """The cleaner uses the same allow-list (.md/.rst/.Rmd/.Rd);
        polish must skip the same set so we don't waste an LLM call on a
        .py file that has nothing to polish."""
        mgr = _make_manager()
        mgr.polisher = MagicMock()
        for fname in ("script.py", "data.txt", "Makefile", "image.png"):
            assert mgr._polish_content(fname, "body") == "body"
        mgr.polisher.polish.assert_not_called()

    def test_runs_on_each_supported_extension(self):
        mgr = _make_manager()
        mgr.polisher = MagicMock()
        mgr.polisher.polish.return_value = ("body", {})
        for fname in ("a.md", "b.rst", "c.Rmd", "d.Rd"):
            mgr._polish_content(fname, "body")
        # One call per extension — the gate didn't accidentally drop any.
        assert mgr.polisher.polish.call_count == 4

    def test_skips_when_polish_output_false(self):
        """The ablation knob must short-circuit BEFORE any polish call —
        otherwise we'd still pay the LLM cost and just ignore the output."""
        mgr = _make_manager(polish_output=False)
        mgr.polisher = MagicMock()
        out = mgr._polish_content("foo.md", "body")
        assert out == "body"
        mgr.polisher.polish.assert_not_called()


class TestPolishContentHappyPath:
    def test_accepts_polished_output_when_guardrail_passes(self):
        """End-to-end through the real ``_accept_polish_if_safe``: an
        identical polish (same length, same fence count, same header count)
        is accepted and returned."""
        mgr = _make_manager()
        mgr.polisher = MagicMock()
        body = "# H\n\nprose\n"
        mgr.polisher.polish.return_value = (body, {"total_tokens": 5})
        out = mgr._polish_content("doc.md", body)
        assert out == body
        mgr.polisher.polish.assert_called_once_with(body)

    def test_falls_back_to_original_when_guardrail_rejects(self):
        """Polish drifted on length (empty output) — guardrail rejects and
        ``_polish_content`` returns the un-polished content rather than
        regressing to empty."""
        mgr = _make_manager()
        mgr.polisher = MagicMock()
        body = "# H\n\nprose body\n"
        mgr.polisher.polish.return_value = ("", {})
        assert mgr._polish_content("doc.md", body) == body

    def test_swallows_polish_exception_and_returns_content(self):
        """LLM failures (rate limits, network) must not break the run —
        same defensive contract as ``_clean_content``."""
        mgr = _make_manager()
        mgr.polisher = MagicMock()
        mgr.polisher.polish.side_effect = RuntimeError("rate limit")
        body = "# H\n\nbody\n"
        assert mgr._polish_content("doc.md", body) == body


# ---------------------------------------------------------------------------
# Render loop — polish runs BEFORE clean
# ---------------------------------------------------------------------------

class TestPolishRunsBeforeCleanInRenderLoop:
    """Reversing the order would let polish-introduced "AI tells" slip
    past the cleaner (the cleaner strips concluding remarks, summaries,
    etc.) and let cleaner-introduced surface drift slip past polish."""

    def _drive_render(self, mgr, *, fpath="doc.md", content_after_edits="# body\n"):
        # Stub out the per-file edit step so we control exactly what
        # _polish_content / _clean_content receive.
        mgr._process_file_edits = MagicMock(
            return_value=(content_after_edits, {"added_lines": 0})
        )
        edits_by_file = {fpath: [MagicMock()]}
        files = {fpath: content_after_edits}
        revised, _ = mgr._render_documents(
            edits_by_file=edits_by_file,
            files=files,
            suggestions=[],
            plan=MagicMock(),
            report=MagicMock(),
        )
        return revised

    def test_polish_called_before_clean_with_correct_inputs(self):
        mgr = _make_manager()
        # Use a shared recorder so we can assert the relative call order
        # without relying on call counts alone.
        order: list[str] = []
        mgr._polish_content = MagicMock(
            side_effect=lambda fp, c: (order.append(f"polish:{c}"), "POLISHED")[1]
        )
        mgr._clean_content = MagicMock(
            side_effect=lambda fp, c: (order.append(f"clean:{c}"), "CLEANED")[1]
        )
        revised = self._drive_render(mgr, content_after_edits="EDITED")
        assert order == ["polish:EDITED", "clean:POLISHED"]
        # The cleaner's output is what lands in ``revised`` — polish feeds
        # clean, clean feeds the dict.  A reversed wiring would surface here
        # as ``revised["doc.md"] == "POLISHED"``.
        assert revised["doc.md"] == "CLEANED"

    def test_polish_receives_edited_content_not_original(self):
        """Subtle: polish must operate on the generator's output, never on
        the un-edited file from disk — otherwise we'd polish away the very
        edits the pipeline just produced."""
        mgr = _make_manager()
        mgr._polish_content = MagicMock(return_value="POLISHED")
        mgr._clean_content = MagicMock(return_value="CLEANED")
        self._drive_render(mgr, content_after_edits="EDITED_BY_PIPELINE")
        # Argument to polish must be the edited content (positional arg 1).
        ((_fp, arg_content),) = [c.args for c in mgr._polish_content.call_args_list]
        assert arg_content == "EDITED_BY_PIPELINE"


# ---------------------------------------------------------------------------
# Source-level guard — polish must not call get_openai()
# ---------------------------------------------------------------------------

class TestPolishContentNoHiddenDefaultLLM:
    def test_polish_content_source_does_not_reference_get_openai(self):
        """Same regression guard as ``_generate_continuation`` /
        ``MarkdownPolisher`` itself.  Source-level so the rule holds
        regardless of import-time sandbox quirks."""
        import inspect
        src = inspect.getsource(DocumentationGenerationManager._polish_content)
        assert "get_openai" not in src
        assert "self.polisher" in src
