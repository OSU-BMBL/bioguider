"""Unit tests for ``bioguider/generation/markdown_polisher.py``.

Two surfaces pinned here:

  * ``MarkdownPolisher.polish`` must drive a ``CommonConversation``
    built from ``self.llm`` (no hidden default-model factory call),
    feed the document through the polish prompt template, and return
    ``(stripped_content, token_usage)``.

  * ``_accept_polish_if_safe`` must reject any polished output that
    drifts on length, fence count, or header count — so a misbehaving
    polish call can never regress structure relative to the
    pre-polish ``refined`` document.

Both are exercised without real LLM calls by patching
``CommonConversation`` at the module level (same pattern as
``test_llm_content_generator_llm.py``).
"""
from __future__ import annotations

import inspect
from unittest.mock import patch

import pytest

from bioguider.generation.markdown_polisher import (
    MarkdownPolisher,
    POLISH_PROMPT,
    _accept_polish_if_safe,
    _count_fences,
    _count_headers,
)


class _SentinelLLM:
    """Stand-in LLM — distinct from anything ``get_openai()`` would return."""
    model_name = "kimi-k2.5"


def _patch_conv(return_content: str = "polished output",
                return_usage: dict | None = None):
    """Patch ``CommonConversation`` so ``polish`` runs without an LLM call.

    Returns ``(stop_callable, MockConv)`` — the caller is responsible for
    invoking ``stop_callable`` in a ``finally`` block.
    """
    usage = return_usage if return_usage is not None else {
        "total_tokens": 11, "prompt_tokens": 5, "completion_tokens": 6,
    }
    p = patch("bioguider.generation.markdown_polisher.CommonConversation")
    MockConv = p.start()
    MockConv.return_value.generate.return_value = (return_content, usage)
    return p.stop, MockConv


# ---------------------------------------------------------------------------
# MarkdownPolisher.polish — LLM is self.llm, never a default
# ---------------------------------------------------------------------------

class TestPolisherUsesPassedLLM:
    def test_builds_conversation_with_self_llm(self):
        llm = _SentinelLLM()
        stop, MockConv = _patch_conv()
        try:
            MarkdownPolisher(llm).polish("# doc\n\nbody.\n")
        finally:
            stop()
        MockConv.assert_called_once_with(llm)

    def test_source_does_not_reference_get_openai(self):
        """Source-level guard mirroring the LLMContentGenerator regression
        test — pins that the polish path never grows a hidden default-model
        fallback that would silently swap the model under test."""
        src = inspect.getsource(MarkdownPolisher)
        assert "get_openai" not in src
        assert "self.llm" in src


# ---------------------------------------------------------------------------
# MarkdownPolisher.polish — return shape and prompt formatting
# ---------------------------------------------------------------------------

class TestPolisherReturnAndPrompt:
    def test_returns_content_and_usage_tuple(self):
        stop, MockConv = _patch_conv(
            return_content="cleaned body",
            return_usage={"total_tokens": 42, "prompt_tokens": 30, "completion_tokens": 12},
        )
        try:
            out, usage = MarkdownPolisher(_SentinelLLM()).polish("# doc\n")
        finally:
            stop()
        assert out == "cleaned body"
        assert usage == {"total_tokens": 42, "prompt_tokens": 30, "completion_tokens": 12}

    def test_strips_outer_whitespace(self):
        stop, _MockConv = _patch_conv(return_content="\n\n   real body   \n\n")
        try:
            out, _ = MarkdownPolisher(_SentinelLLM()).polish("# doc\n")
        finally:
            stop()
        assert out == "real body"

    def test_system_prompt_carries_document_body_inline(self):
        stop, MockConv = _patch_conv()
        try:
            MarkdownPolisher(_SentinelLLM()).polish("# UniqueAnchor99\n\nbody.\n")
        finally:
            stop()
        kwargs = MockConv.return_value.generate.call_args.kwargs
        # The polish prompt template is the system prompt, with the document
        # spliced in.  Asserting both halves so a refactor that drops either
        # surface (template OR substitution) trips this test.
        assert "polishing the FINAL surface" in kwargs["system_prompt"]
        assert "# UniqueAnchor99" in kwargs["system_prompt"]
        # Instruction prompt is fixed wording — pinning so prompt drift
        # surfaces immediately, not silently changes per-model behaviour.
        assert kwargs["instruction_prompt"].startswith("Return the polished")

    def test_long_document_is_truncated_to_30000_chars_in_prompt(self):
        """The prompt cap (30_000 chars) protects context windows and keeps
        polish cost bounded; the call must NOT include the full long body."""
        long_doc = "x" * 50_000
        stop, MockConv = _patch_conv()
        try:
            MarkdownPolisher(_SentinelLLM()).polish(long_doc)
        finally:
            stop()
        prompt = MockConv.return_value.generate.call_args.kwargs["system_prompt"]
        # The full 50k body must not be in the prompt; the 30k slice must.
        assert "x" * 50_000 not in prompt
        assert "x" * 30_000 in prompt


# ---------------------------------------------------------------------------
# POLISH_PROMPT contract — pin the rules the guardrail can't enforce
# ---------------------------------------------------------------------------

class TestPolishPromptContract:
    """The polish prompt is the only thing keeping the LLM out of fenced
    blocks, off identifiers, and away from restructuring.  Each rule below
    is load-bearing for either correctness (don't undo pipeline fixes) or
    for the §5 guardrail to remain a fallback rather than the primary
    defense.  Treat changes to these strings as API changes."""

    def test_forbids_modifying_fenced_blocks(self):
        assert "DO NOT modify anything inside fenced code blocks" in POLISH_PROMPT

    def test_forbids_changing_cli_flag_or_identifier_names(self):
        assert "DO NOT change CLI flag names, function names, file names" in POLISH_PROMPT

    def test_forbids_structural_edits(self):
        assert "DO NOT add, remove, reorder, or retitle sections" in POLISH_PROMPT

    def test_forbids_concluding_remarks(self):
        # ``LLMCleaner`` strips these too, but polish runs first — if polish
        # adds an "AI tell" we'd rather the prompt forbid it up front than
        # rely on cleaner to remove it later.
        assert "DO NOT add concluding remarks" in POLISH_PROMPT


# ---------------------------------------------------------------------------
# _count_fences / _count_headers — small helpers, easy to get wrong
# ---------------------------------------------------------------------------

class TestFenceAndHeaderCounters:
    def test_count_fences_matches_line_starts_only(self):
        # A backtick sequence inline in prose must not count as a fence;
        # only line-start ``` delimiters do.
        text = "Use ```code``` inline.\n```\nfenced\n```\n"
        assert _count_fences(text) == 2

    def test_count_fences_zero_for_no_fences(self):
        assert _count_fences("just prose\n") == 0

    def test_count_headers_counts_atx_only(self):
        text = "# A\n## B\n### C\nnot a header\n#nospace\n"
        # The fourth line has no leading # at all; the fifth has no space
        # after # and is intentionally NOT counted by the regex.
        assert _count_headers(text) == 3

    def test_count_headers_ignores_inline_hash(self):
        text = "Text with # in the middle\n"
        assert _count_headers(text) == 0


# ---------------------------------------------------------------------------
# _accept_polish_if_safe — guardrail
# ---------------------------------------------------------------------------

# Long enough that a single structural removal (~12 chars for a header,
# ~14 chars for a fence pair's wrapper) stays well within the ±10% length
# band — so the relevant guardrail (fence/header count) is what trips,
# not the length check.  Padding is plain prose so it can't accidentally
# affect either counter.
_PROSE_PAD = ("Filler paragraph that exists only to give the fixture enough "
              "length that small structural deletions stay within the 10%% "
              "length-tolerance band of the polish guardrail.\n\n") * 6

REFINED = (
    "# Title\n"
    "\n"
    + _PROSE_PAD +
    "Some prose with `inline` code.\n"
    "\n"
    "## Section\n"
    "\n"
    + _PROSE_PAD +
    "```bash\n"
    "python tool.py --flag value\n"
    "```\n"
    "\n"
    + _PROSE_PAD +
    "More prose.\n"
)


def _polish_keeping_structure(refined: str, change: str = "") -> str:
    """Polished text whose surface differs from ``refined`` but whose
    fence-count, header-count, and length stay within tolerance."""
    # Touch only prose; preserve every newline and structural marker.
    return refined.replace("Some prose", "Some prose" + change)


class TestAcceptPolishIfSafe:
    def test_accepts_polished_when_structure_preserved(self):
        polished = _polish_keeping_structure(REFINED)  # identical
        assert _accept_polish_if_safe(REFINED, polished, REFINED) is polished

    def test_accepts_minor_prose_edits_within_length_tolerance(self):
        polished = _polish_keeping_structure(REFINED, change=" (polished)")
        out = _accept_polish_if_safe(REFINED, polished, REFINED)
        assert out is polished

    def test_rejects_empty_polish(self):
        assert _accept_polish_if_safe(REFINED, "", REFINED) == REFINED

    def test_rejects_when_polish_shrinks_below_90_percent(self):
        # 50% the original length: too short, fall back.
        polished = REFINED[: len(REFINED) // 2]
        assert _accept_polish_if_safe(REFINED, polished, REFINED) == REFINED

    def test_rejects_when_polish_grows_above_110_percent(self):
        # 200% original length: model rewrote far more than asked.
        polished = REFINED + REFINED
        assert _accept_polish_if_safe(REFINED, polished, REFINED) == REFINED

    def test_rejects_when_fence_count_diverges(self):
        # Polish converted the bash fence into prose.  Fence count drops
        # by 2; length stays well within ±10% (only the two fence lines
        # are removed).  Guardrail must catch this — losing the fence
        # would be a silent structural regression.
        polished = REFINED.replace(
            "```bash\npython tool.py --flag value\n```\n",
            "python tool.py --flag value\n",
        )
        assert _count_fences(polished) != _count_fences(REFINED)
        # Confirm the gating: length within tolerance, headers unchanged,
        # so fence drift is what actually trips the guardrail.
        assert 0.9 <= len(polished) / len(REFINED) <= 1.1
        assert _count_headers(polished) == _count_headers(REFINED)
        assert _accept_polish_if_safe(REFINED, polished, REFINED) == REFINED

    def test_rejects_when_header_count_diverges(self):
        # Polish dropped the ## Section header — structural drift.
        polished = REFINED.replace("## Section\n\n", "")
        assert _count_headers(polished) != _count_headers(REFINED)
        # Confirm the gating: length within tolerance, fences unchanged, so
        # header drift is what trips the guardrail.
        assert 0.9 <= len(polished) / len(REFINED) <= 1.1
        assert _count_fences(polished) == _count_fences(REFINED)
        assert _accept_polish_if_safe(REFINED, polished, REFINED) == REFINED

    def test_does_not_divide_by_zero_on_empty_refined(self):
        # Edge case: pre-polish content is empty (generator returned "").
        # Guardrail must not blow up with ZeroDivisionError.
        assert _accept_polish_if_safe("", "anything", "") == "anything" or \
               _accept_polish_if_safe("", "anything", "") == ""
        # Empty refined → ratio = len(polished)/1 = len(polished); a non-empty
        # polish (8 chars) is far above 1.1, so guardrail rejects.
        assert _accept_polish_if_safe("", "anything", "") == ""

    def test_original_argument_is_accepted_but_currently_ignored(self):
        """The signature carries ``original`` for future stricter rules
        (anchoring section count to the unmutated source).  Today it must
        not affect the decision — verifying so a future change is a
        deliberate API tightening, not an accident."""
        polished = _polish_keeping_structure(REFINED, change=" (touch)")
        out1 = _accept_polish_if_safe(REFINED, polished, REFINED)
        out2 = _accept_polish_if_safe(REFINED, polished, "totally different original")
        assert out1 == out2
