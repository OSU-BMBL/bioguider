"""Surface-markdown polish pass for the generation pipeline.

This runs AFTER the generator has produced a substantively-repaired
document.  Its only job is to clean up a narrow class of residual
surface errors — broken inline-code spans, mis-formed image / link
syntax, and obvious prose typos — that the evaluation tasks do not
emit explicit suggestions for and that the generator therefore tends
to leave untouched.

Why it exists
-------------
Per the benchmark analysis on pharokka, the pipeline strategy already
beats the bare-prompt ``simple`` strategy on the categories where
repo-aware evaluation matters (notably ``cli_unknown_flag``, +46.9 pp).
But it loses ground on surface-markdown categories — most notably
``inline_code`` (−9.9 pp) — because:

  * Evaluation tasks emit high-level categories
    (``readability``/``setup``/``structure``/...) and never produce a
    targeted ``inline_code`` finding.
  * ``LLMContentGenerator`` preserves structure faithfully, so
    untargeted surface defects survive the rewrite.
  * The ``simple`` strategy's "fix all errors" disposition picks them
    up naturally during reproduction.

Adding a tightly-scoped polish pass closes this gap without giving up
the pipeline's domain wins.

Contract
--------
- Uses ``self.llm`` end-to-end; never calls a default-model factory.
  (Same invariant we just enforced in ``LLMContentGenerator``.)
- Returns ``(polished_content, token_usage)``.  Callers MUST gate the
  swap-in on ``_accept_polish_if_safe`` so a misbehaving polish call
  cannot regress structure or length.
"""
from __future__ import annotations

import re

from langchain_openai.chat_models.base import BaseChatOpenAI

from bioguider.agents.common_conversation import CommonConversation


# Hard cap on what we feed to the polisher.  Matches LLMCleaner's cap so the
# two surface-passes can coexist without separate truncation policies.
_MAX_PROMPT_DOC_CHARS = 30000


POLISH_PROMPT = """You are polishing the FINAL surface of a documentation file
that has already been substantively repaired by an upstream pipeline.

YOUR ONLY JOB is to fix four narrow classes of remaining surface errors:
  1. Broken inline code spans (mismatched or missing backticks)
  2. Broken image syntax outside fences:  ![alt](url)
  3. Broken link syntax outside fences:   [text](url)
  4. Obvious prose typos in ordinary English words

STRICT RULES — violations are bugs:
  - DO NOT modify anything inside fenced code blocks (``` ... ```)
  - DO NOT modify YAML frontmatter
  - DO NOT change CLI flag names, function names, file names, or any
    identifier — those were intentionally repaired upstream
  - DO NOT add, remove, reorder, or retitle sections
  - DO NOT add concluding remarks, summaries, or "happy analyzing"
  - Output the COMPLETE polished document, nothing else

INPUT
<<DOCUMENT>>
{doc}
<</DOCUMENT>>

Return ONLY the polished document content (no commentary, no fences)."""


class MarkdownPolisher:
    """Run one LLM-driven surface polish on a generated document.

    Construct with the same LLM the generator was given so the polish
    pass is attributed to the model under test (critical for per-model
    benchmark comparisons — a hard-coded fallback would silently swap
    the model and contaminate results).
    """

    def __init__(self, llm: BaseChatOpenAI):
        self.llm = llm

    def polish(self, content: str) -> tuple[str, dict]:
        """Polish ``content`` and return ``(polished, token_usage)``.

        Returns the stripped LLM output verbatim — caller is responsible
        for accepting it only after ``_accept_polish_if_safe`` passes.
        """
        conv = CommonConversation(self.llm)
        output, token_usage = conv.generate(
            system_prompt=POLISH_PROMPT.format(doc=content[:_MAX_PROMPT_DOC_CHARS]),
            instruction_prompt="Return the polished document content only.",
        )
        return output.strip(), token_usage


# ---------------------------------------------------------------------------
# Guardrail — pure-Python sanity checks on the polished output.
# ---------------------------------------------------------------------------

# A line that opens or closes a fenced code block.  Match the same anchor
# the injector / metrics use so the polish guardrail is consistent with the
# rest of the pipeline.
_FENCE_RE = re.compile(r"^```", re.MULTILINE)

# ATX-style header line (``# Title`` … ``###### Title``).  Setext headers
# (underlined with ``===`` / ``---``) are intentionally not counted — we only
# care about structural drift and ATX is what the generator emits.
_HEADER_RE = re.compile(r"^#{1,6}\s+\S", re.MULTILINE)

# Tolerated length drift between the polished and refined documents.  Polish
# should be near-idempotent on length; anything outside this band signals a
# rewrite, not a polish.
_LENGTH_TOLERANCE = 0.10


def _count_fences(text: str) -> int:
    return len(_FENCE_RE.findall(text))


def _count_headers(text: str) -> int:
    return len(_HEADER_RE.findall(text))


def _accept_polish_if_safe(refined: str, polished: str, original: str) -> str:
    """Return ``polished`` if it passes structural guardrails, else ``refined``.

    Worst case is "no improvement" — we never regress relative to the
    pre-polish output.  The ``original`` argument is accepted for symmetry
    with the caller's signature and to leave room for stricter checks
    later (e.g. anchoring section count to the original), but is not used
    by the current rules.
    """
    # Empty / missing output — polish produced nothing usable.
    if not polished:
        return refined

    refined_len = max(len(refined), 1)
    ratio = len(polished) / refined_len
    if not (1 - _LENGTH_TOLERANCE <= ratio <= 1 + _LENGTH_TOLERANCE):
        return refined

    # Fence count parity: polish must not open or close code blocks.
    if _count_fences(polished) != _count_fences(refined):
        return refined

    # Header count parity: polish must not restructure sections.
    if _count_headers(polished) != _count_headers(refined):
        return refined

    return polished
