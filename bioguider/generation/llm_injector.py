from __future__ import annotations

import json
from typing import Tuple, Dict, Any, List, Optional, Set
import re
from difflib import SequenceMatcher

from langchain_openai.chat_models.base import BaseChatOpenAI
from bioguider.agents.common_conversation import CommonConversation
from bioguider.utils.utils import escape_braces


INJECTION_PROMPT = """
You are “BioGuider-Intro,” generating a deliberately flawed **INTRODUCTION** file
(“README-lite”) to test an auto-fixer. Start from the provided clean INTRO doc that follows the
BioGuider Intro structure (What is it? / What can it do? / Requirements / Install / Quick example /
Learn more / License & Contact). Produce a corrupted version with small, realistic defects.

GOAL
Introduce subtle but meaningful issues while keeping the document recognizably the same.

ERROR CATEGORIES (inject all)
- typo: spelling/grammar/punctuation mistakes
- link: malformed URL, wrong domain, or stray spaces in URL
- duplicate: duplicate a short line/section fragment
- bio_term: slightly wrong domain term (e.g., “single sell” for “single cell”); do not invent new science
- function: misspell a known function/API name **from the input README-lite only**
- markdown_structure: break a header level, list indentation, or code fence (one-off)
- list_structure: remove bullet space (e.g., “-item”), mix markers inconsistently
- section_title: subtly change a section title casing or wording
- image_syntax: break image markdown spacing (e.g., `![alt] (url)`)
- inline_code: remove backticks around inline code
- emphasis: break emphasis markers (e.g., missing closing `*`)
- table_alignment: misalign or omit a `|` in a markdown table
- code_lang_tag: use the wrong fenced code language (e.g., ```py for R)

BIOLOGY-SPECIFIC ERROR CATEGORIES (inject all; keep realistic & subtle)
- gene_symbol_case: change gene symbol casing or add suffix (e.g., “tp53”, “CD3e”), but **do not alter** protected keywords
- species_swap: imply human vs mouse mix-up (e.g., “mm10” vs “GRCh38”) in a short phrase
- ref_genome_mismatch: claim a reference genome that conflicts with the example file or text
- modality_confusion: conflate RNA-seq with ATAC or proteomics in a brief phrase
- normalization_error: misuse terms like CPM/TPM/CLR/log1p in a sentence
- umi_vs_read: confuse UMI counts vs read counts in a short line
- batch_effect: misstate “batch correction” vs “normalization” terminology
- qc_threshold: use a common but slightly wrong QC gate (e.g., mito% 0.5 instead of 5)
- file_format: mix up FASTQ/BAM/MTX/H5AD/RDS in a brief mention
- strandedness: claim “stranded” when workflow is unstranded (or vice versa)
- coordinates: confuse 0-based vs 1-based or chromosome naming style (chr1 vs 1)
- units_scale: use the wrong scale/unit (e.g., μm vs mm; 10e6 instead of 1e6)
- sample_type: conflate “primary tissue” with “cell line” in a single phrase
- contamination: misuse “ambient RNA” vs “doublets” terminology

CLI/CONFIG ERROR CATEGORIES (inject all)
- param_name: slightly misspell a CLI flag or config key (e.g., `--min-cell` → `--min-cells`)
- default_value: state a plausible but incorrect default value
- path_hint: introduce a subtle path typo (e.g., `data/filtrd`)
- cli_flag_typo: permute (transpose two adjacent characters in) a long CLI flag so the parser would not define it (e.g., `--size` → `--szie`, `--epochs` → `--epohcs`). When the flag appears in multiple places (fenced blocks, inline `` `…` `` spans, help/usage stanzas, prose option references), mutate EVERY occurrence consistently so the document stays self-consistent in its wrongness. Do not break fence delimiters or inline-code backticks.
- cli_unknown_flag: inside a fenced code block, append a plausible-looking bogus long flag that the program's parser does not define (e.g., `python train.py --epochs 20` → `python train.py --epochs 20 --workers 4`). Inject ONLY inside fenced code blocks; never break fence delimiters.
- cli_program_rename: inside a fenced code block, mutate the script-name token of a shell-style invocation so the file does not exist in the repo (e.g., `python scripts/train.py ...` → `python scripts/trian.py ...`). Inject ONLY inside fenced code blocks; do not modify the flags or values; never break fence delimiters.

EXTERNAL-AUTHORITY ERROR CATEGORIES (inject only when a context anchor exists)
- accession_id_prefix: swap an accession ID prefix so the namespace contradicts a nearby context word (e.g., prose says "series" but mutate `GSE123456` → `GSM123456`, or prose says "samples" but mutate `GSM123456` → `GSE123456`). Only inject when a context word (series/samples/experiment/run/study) appears within the same sentence as the accession ID. Otherwise skip and record in "skipped".

PROSE-CODE CONSISTENCY ERROR CATEGORIES (inject ONLY when a matching code-block anchor exists; otherwise skip and record in "skipped" with reason "no_anchor")
- prose_code_pkg_version: prose narrates a package major version that disagrees with the version pinned/loaded in a fenced code block (e.g., `library(Seurat)` with `sessionInfo()` showing `Seurat_5.0.1`, but prose says "Seurat v4"). Mutate the prose version only; leave code untouched.
- prose_code_stat_test: prose narrates a statistical test name that disagrees with the function actually called in a fenced code block (e.g., code runs `wilcox.test(...)` or `FindMarkers(..., test.use="wilcox")`, but prose says "t-test"). Mutate the prose test name only.
- prose_code_marker: prose names a cell-type marker gene that disagrees with the marker used in a fenced code block (e.g., code subsets on `CD8` via `subset(..., CD8 > 0)` or `features = c("CD8")`, but prose describes "CD4+ T cells"). Mutate the prose marker only.
- prose_code_param: prose states an analysis hyperparameter value that disagrees with the value passed to the corresponding function call in a fenced code block (e.g., code runs `FindClusters(..., resolution = 0.5)`, but prose says "resolution of 0.6"). Mutate the prose value only.

CODE CONSISTENCY ERROR CATEGORIES (inject all; ONLY inside fenced code blocks — never break fence delimiters)
- code_comment_conflict: change an inline code comment (a line starting with #) inside a code block so it conflicts with the function called on the same or the next line (e.g., change "# normalize the data" to "# cluster the data" when the code calls NormalizeData()). Use antonym/opposite-operation replacements only; keep the comment grammatically correct.

CONSTRAINTS
- Keep edits minimal and local; **≥85% token overlap** with input.
- **CRITICAL: Preserve ALL code block structure exactly**:
  * Do NOT remove, add, or modify code fence delimiters (``` or ```{r} or ```{python})
  * The number of ``` lines MUST be identical in input and output
  * For RMarkdown/Rmd files, preserve ALL chunk headers like ```{r, ...}
  * Only introduce errors INSIDE code blocks (typos in code), never break the fences
- **Preserve section ORDER and TITLES** from the Intro spec (if applicable):
  1) # <project_name>
     _<tagline>_
  2) What is it?
  3) What can it do?
  4) Requirements
  5) Install
  6) Quick example
  7) Learn more
  8) License & Contact
- Do **not** add or remove top-level sections. Subtle line-level corruption only.
- Maintain a **concise length** (≤ {max_words} words).
- Do **not** alter the protected keywords (exact casing/spelling): {keywords}
- Keep at least **{min_per_category} errors per category** listed above.
- Limit `duplicate` injections to at most **{min_per_category}**.
- If the input contains runnable code, keep it mostly intact but introduce **one** realistic break
  (e.g., missing quote/paren or wrong function name) without adding new libraries.
- Keep at least one **valid** URL so the fixer can compare.
- Do not change the project identity, domain, or language.
- Do not include markers, explanations, or commentary in the corrupted markdown.

INPUT INTRO (clean README-lite)
<<INTRO>>
{readme}
<</INTRO>>

OUTPUT (JSON only):
{{
  "corrupted_markdown": "<the entire corrupted INTRO as markdown>",
  "errors": [
    {{
      "id": "e1",
      "category": "typo|link|duplicate|bio_term|function|markdown_structure",
      "rationale": "why this mutation is realistic",
      "original_snippet": "<verbatim snippet from input>",
      "mutated_snippet": "<verbatim mutated text>"
    }}
    // include one entry per individual mutation you applied
  ]
}}
"""


_CODE_FENCE_RE = re.compile(r"```[^\n]*\n.*?```", re.DOTALL)
_PKG_ANCHOR_RE = re.compile(
    r"\b(Seurat|scanpy|SingleCellExperiment|DESeq2|edgeR|limma)[_=\s]+v?(\d+)(?:\.\d+)*",
    re.I,
)
_STAT_TEST_ANCHOR_RE = re.compile(
    r"\bwilcox(?:\.test|on)?\s*\(|"
    r'test\.use\s*=\s*["\']wilcox["\']|'
    r"\bt\.test\s*\(|"
    r'test\.use\s*=\s*["\']t["\']|'
    r"\bFind(?:All)?Markers\s*\(",   # Seurat idiom: default test.use is "wilcox"
    re.I,
)
_MARKER_ANCHOR_RE = re.compile(
    r'(?:features?\s*=\s*(?:c\(|\[)?\s*["\']([A-Z][A-Z0-9]{1,6})["\']|'
    r"subset\([^)]*?\b([A-Z][A-Z0-9]{1,6})\s*[><=]|"
    # Seurat plotting idioms — marker gene appears as positional string arg
    r'(?:FeaturePlot|VlnPlot|DotPlot|RidgePlot)\s*\([^,)]*,\s*(?:features?\s*=\s*)?["\']([A-Z][A-Z0-9]{1,6})["\'])'
)
_PARAM_ANCHOR_RE = re.compile(
    r"(resolution|n[._]neighbors|perplexity|n[._]components|min[._]dist|spread)"
    r"\s*=\s*(\d+\.?\d*)",
    re.I,
)
_ACCESSION_CONTEXT_WORDS = ("series", "samples", "experiment", "run", "study", "geo")

# ── Code-consistency injection helpers ──────────────────────────────────────
# Function/class calls inside code blocks (name must be ≥5 chars to avoid
# corrupting short base-language tokens like `if(`, `c(`, `do(`).
_FUNC_IN_CODE_RE = re.compile(r"\b([A-Za-z_][A-Za-z0-9_.]{4,})\s*\(", re.M)
# Named arguments: `arg_name = ` (≥3 chars)
_NAMED_ARG_IN_CODE_RE = re.compile(r"\b([a-z_][a-z0-9_.]{2,})\s*=\s*", re.M)
# Inline comment lines inside code blocks
_COMMENT_IN_CODE_RE = re.compile(r"^([ \t]*#[ \t]+)(.+)$", re.M)

# Tokens that must never be mutated (R control-flow, common base functions)
_CODE_SKIP_NAMES: set = {
    "function", "return", "print", "paste", "paste0", "list", "data",
    "library", "require", "source", "if", "for", "while", "repeat",
    "switch", "stop", "warning", "message", "class", "is", "as",
    "tryCatch", "withCallingHandlers",
}

# ── .Rd documentation injection helpers ─────────────────────────────────────
# Sections whose content must NOT be mutated (executable / machine-generated).
_RD_SKIP_SECTIONS = (r"\usage", r"\examples")

# \code{FuncName()} or \code{FuncName} in prose — function reference
_RD_CODE_FUNC_RE = re.compile(r"\\code\{([A-Za-z][A-Za-z0-9._]{3,})\(?[^}]*\}", re.M)
# \code{argname} in prose — likely an argument reference (no parens, lowercase-ish)
_RD_CODE_ARG_RE  = re.compile(r"\\code\{([a-z][a-z0-9._]{2,})\}", re.M)
# \link[pkg]{FuncName} or \link{FuncName} cross-references
_RD_LINK_RE      = re.compile(r"\\link(?:\[[^\]]*\])?\{([A-Za-z][A-Za-z0-9._]{3,})\}", re.M)

# Semantic-conflict replacement map for comment injection
_COMMENT_CONFLICT_MAP = [
    (re.compile(r"\bload\b", re.I), "save"),
    (re.compile(r"\bfilter\b", re.I), "cluster"),
    (re.compile(r"\bnormali[sz]e\b", re.I), "scale"),
    (re.compile(r"\bscale\b", re.I), "normalize"),
    (re.compile(r"\bcluster\b", re.I), "normalize"),
    (re.compile(r"\bplot\b|\bvisuali[sz]e\b", re.I), "cluster"),
    (re.compile(r"\bmerge\b", re.I), "split"),
    (re.compile(r"\bsplit\b", re.I), "merge"),
    (re.compile(r"\btrain\b", re.I), "predict"),
    (re.compile(r"\bpredict\b", re.I), "train"),
    (re.compile(r"\bintegrat\w*\b", re.I), "subset"),
    (re.compile(r"\bsubset\b", re.I), "integrate"),
]

# ── File-type gates for code-consistency injection ───────────────────────────
# Plain-text docs where function/arg names in prose can be verified and fixed.
_PROSE_CODE_EXTENSIONS: frozenset = frozenset({".md", ".rst", ".txt"})
# Executable notebooks — never mangle code inside them.
_EXECUTABLE_CODE_EXTENSIONS: frozenset = frozenset({".rmd", ".ipynb"})

# ── CLI-consistency injection helpers ───────────────────────────────────────
# Fence languages whose contents are treated as raw shell invocations.
_CLI_SHELL_LANGS: frozenset = frozenset(
    {"bash", "sh", "shell", "console", "terminal", "zsh"}
)
# Tokens that signal "this line invokes a Python or R interpreter".
_CLI_LAUNCHERS: tuple = ("python3", "python", "py", "Rscript")
# Long flags we must never mutate (well-known across nearly every CLI).
_CLI_PROTECTED_FLAGS: frozenset = frozenset(
    {"--help", "--version", "--verbose", "--quiet"}
)
# Deterministic round-robin source for cli_unknown_flag injections.
_CLI_BOGUS_FLAGS: tuple = (
    ("--workers", "4"),
    ("--nproc", "8"),
    ("--threads", "2"),
    ("--cores", "4"),
)
_CLI_SCRIPT_SUFFIXES: tuple = (".py", ".R", ".r")
# Argparse stanza headers ("usage:", "options:", "positional arguments:",
# "optional arguments:") — these label help output, not command lines.
_CLI_USAGE_HEADER_RE = re.compile(
    r"^\s*(usage:|options:|positional\s+arguments:|optional\s+arguments:)",
    re.I,
)

# Argparse-style ``usage: prog [-h] [--flag VAL] …`` block.  The first line
# starts with ``usage:``; continuation lines are indented (argparse wraps
# long usage lines and aligns the continuation to the character after
# ``usage: ``).  The block ends at the first non-indented line — typically
# a blank line before the program description.
_HELP_USAGE_BLOCK_RE = re.compile(
    r"^[ \t]*usage:[^\n]*(?:\n[ \t]+[^\n]*)*",
    re.M,
)
# Argparse ``options:`` / ``optional arguments:`` section.  Header line is
# captured separately from the indented body; ``re.I`` because some tools
# emit ``Options:`` with a capital O.
_HELP_OPTIONS_BLOCK_RE = re.compile(
    r"(?P<head>^[ \t]*(?:options|optional[ \t]+arguments):[ \t]*\n)"
    r"(?P<body>(?:[ \t]+[^\n]*\n)+)",
    re.M | re.I,
)
# Single-backtick inline-code span (one line only). Negative lookarounds
# exclude double-backtick spans `` `text with ` backtick` `` ``, which are
# rare and need different scanning.
_INLINE_CODE_RE = re.compile(r"(?<!`)`([^`\n]+)`(?!`)")


class LLMErrorInjector:
    def __init__(self, llm: BaseChatOpenAI, force_deterministic: bool = False):
        """
        Args:
            llm: Language model used for the (non-deterministic) injection path.
            force_deterministic: When True, skip the LLM call entirely and run
                ``_deterministic_inject`` + ``_supplement_errors`` only. This
                makes injection byte-identical across runs with the same seed,
                which is required for fair cross-model benchmarks (each model
                must see the same corrupted ground truth).
        """
        self.llm = llm
        self.force_deterministic = force_deterministic

    @staticmethod
    def _extract_code_fragments(text: str) -> str:
        """Return concatenated contents of fenced code blocks (the code-authority region)."""
        return "\n".join(
            m.group(0) for m in _CODE_FENCE_RE.finditer(text)
        )

    @staticmethod
    def _prose_region(text: str) -> str:
        """Return text with fenced code blocks stripped (the prose-only region)."""
        return _CODE_FENCE_RE.sub("", text)

    @staticmethod
    def _fence_spans(text: str) -> List[Tuple[int, int]]:
        """Return sorted list of (start, end) character spans for fenced code blocks."""
        return [(m.start(), m.end()) for m in _CODE_FENCE_RE.finditer(text)]

    @staticmethod
    def _yaml_frontmatter_span(text: str) -> "Tuple[int, int] | None":
        """Return (0, end) span of YAML frontmatter if present, else None."""
        if not text.startswith("---"):
            return None
        m = re.search(r"\n---[ \t]*(?:\n|$)", text)
        if m is None:
            return None
        return (0, m.end())

    @staticmethod
    def _in_fence(pos: int, spans: List[Tuple[int, int]]) -> bool:
        """Return True if character position ``pos`` falls inside any fence span."""
        for start, end in spans:
            if start <= pos < end:
                return True
            if start > pos:
                break
        return False

    @staticmethod
    def _replace_in_fence(text: str, old: str, new: str, spans: List[Tuple[int, int]]) -> str:
        """Replace the first occurrence of *old* that IS inside any fence span."""
        start = 0
        while True:
            idx = text.find(old, start)
            if idx == -1:
                return text
            for fstart, fend in spans:
                if fstart <= idx < fend:
                    return text[:idx] + new + text[idx + len(old):]
                if fstart > idx:
                    break
            start = idx + 1

    @staticmethod
    def _replace_prose_only(text: str, old: str, new: str, spans: List[Tuple[int, int]]) -> str:
        """Replace the first occurrence of ``old`` that is NOT inside a fence span."""
        start = 0
        while True:
            idx = text.find(old, start)
            if idx == -1:
                return text  # no prose occurrence found — leave unchanged
            if not LLMErrorInjector._in_fence(idx, spans):
                return text[:idx] + new + text[idx + len(old):]
            start = idx + 1

    # ── .Rd helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _is_rd_file(text: str) -> bool:
        """Return True when text looks like an .Rd R-documentation file."""
        return r"\name{" in text and r"\arguments{" in text

    @staticmethod
    def _rd_skip_spans(text: str) -> List[Tuple[int, int]]:
        """Return (start, end) spans for \\usage{} and \\examples{} blocks.

        Uses brace-depth counting so nested braces inside the block are handled
        correctly.  These spans must never be mutated by the injector.
        """
        spans: List[Tuple[int, int]] = []
        for kw in _RD_SKIP_SECTIONS:
            search_kw = kw + "{"
            pos = 0
            while True:
                idx = text.find(search_kw, pos)
                if idx == -1:
                    break
                # Walk from the opening brace counting depth
                depth = 0
                end = idx
                for i in range(idx + len(kw), len(text)):
                    if text[i] == "{":
                        depth += 1
                    elif text[i] == "}":
                        depth -= 1
                        if depth == 0:
                            end = i + 1
                            break
                spans.append((idx, end))
                pos = end
        spans.sort()
        return spans

    @staticmethod
    def _replace_in_rd_prose(
        text: str, old: str, new: str, skip_spans: List[Tuple[int, int]]
    ) -> str:
        """Replace first occurrence of *old* that is NOT inside any skip span."""
        start = 0
        while True:
            idx = text.find(old, start)
            if idx == -1:
                return text
            in_skip = any(s <= idx < e for s, e in skip_spans)
            if not in_skip:
                return text[:idx] + new + text[idx + len(old):]
            start = idx + 1

    @staticmethod
    def _transpose_name(name: str) -> str:
        """Transpose two adjacent characters near the middle of *name*.

        Returns the mutated name, or the original if no valid transposition
        exists (e.g. the name is too short or all adjacent pairs are identical).
        """
        mid = len(name) // 2
        for offset in range(len(name) - 2):
            i = (mid + offset) % (len(name) - 1)
            if name[i] != name[i + 1]:
                return name[:i] + name[i + 1] + name[i] + name[i + 2:]
        return name  # no valid transposition found

    @staticmethod
    def _fence_language(fence_block: str) -> str:
        """Return the lower-cased language tag of a fence block; "" if absent.

        Handles plain `````bash`` and R-markdown chunk headers like
        `````{r, eval=FALSE}``.
        """
        first_nl = fence_block.find("\n")
        opening = fence_block[:first_nl] if first_nl > 0 else fence_block
        tag = opening.lstrip("`").strip()
        if tag.startswith("{"):
            inner = tag[1:tag.find("}")] if "}" in tag else tag[1:]
            return inner.split(",")[0].strip().lower()
        return tag.lower()

    @staticmethod
    def _looks_like_cli_line(line: str, fence_lang: str) -> bool:
        """Return True if *line* is a shell-style invocation worth mutating.

        The gate is intentionally tight: argparse help output frequently lives
        inside ``bash`` fences (usage stanzas, option columns, free-form
        description text) and looked like a CLI invocation under earlier
        rules.  We require an explicit launcher (``python``/``Rscript``/``$``)
        or a ``.py``/``.R``/``.r`` script token, and reject lines that match
        common help-output shapes.
        """
        body = line.strip()
        if not body or body.startswith("#"):
            return False
        # Argparse stanza headers (``usage:``, ``options:``, ...) — labels, not commands.
        if _CLI_USAGE_HEADER_RE.match(line):
            return False
        # Help-text references — clearly explanatory, not invocations.
        if "--help" in body:
            return False
        for p in ("$ ", "> "):
            if body.startswith(p):
                body = body[len(p):].lstrip()
                break
        tokens = body.split()
        if not tokens:
            return False
        first = tokens[0]
        # Argparse argument-list rows start with ``-`` (``-i INFILE, --infile INFILE``)
        # or are usage-stanza continuations starting with ``[`` (``[--foo]``).
        if first[0] in ("-", "["):
            return False
        if first in _CLI_LAUNCHERS or first == "R":
            return True
        return first.endswith(_CLI_SCRIPT_SUFFIXES)

    def _cli_fence_candidates(self, text: str) -> List[Tuple[str, str]]:
        """Return shell-style invocation lines inside fenced code blocks in scan order."""
        results: List[Tuple[str, str]] = []
        for m in _CODE_FENCE_RE.finditer(text):
            block = m.group(0)
            first_nl = block.find("\n")
            if first_nl < 0:
                continue
            lang = self._fence_language(block)
            body = block[first_nl + 1: -3]
            for raw_line in body.splitlines():
                if self._looks_like_cli_line(raw_line, lang):
                    results.append((raw_line, lang))
        return results

    def _cli_inline_candidates(self, text: str) -> List[Tuple[str, str]]:
        """Return CLI invocations that live inside single-backtick inline code.

        pharokka and many similar docs put their command-line examples
        inline (e.g. `` `script.py -i in.fasta --foo bar` `` ) rather than in
        fenced blocks.  Each match is returned as ``(content, "inline")``
        where ``content`` is the text *between* the backticks (no backticks).
        Matches that fall inside a triple-backtick fence are skipped so we
        don't double-count.
        """
        fence_spans = self._fence_spans(text)
        results: List[Tuple[str, str]] = []
        for m in _INLINE_CODE_RE.finditer(text):
            if any(s <= m.start() < e for s, e in fence_spans):
                continue
            content = m.group(1)
            if self._looks_like_cli_line(content, ""):
                results.append((content, "inline"))
        return results

    def _cli_candidates(self, text: str) -> List[Tuple[str, str]]:
        """Return all CLI candidates with a source tag.

        Each entry is ``(line, source)`` where ``source`` is ``"fence"`` for
        a triple-backtick fence line or ``"inline"`` for a single-backtick
        inline-code span.  Fence candidates come first so the historical
        scan order is preserved; new inline candidates are appended.
        """
        out: List[Tuple[str, str]] = [
            (line, "fence") for line, _lang in self._cli_fence_candidates(text)
        ]
        out.extend(self._cli_inline_candidates(text))
        return out

    @staticmethod
    def _replace_in_inline_code(text: str, content: str, mut_content: str) -> str:
        """Replace the first inline-code span whose content equals ``content``.

        Looks for the literal ``\\`content\\``` string (with the surrounding
        backticks), which makes the substitution unambiguous — prose
        without backticks won't match.  Returns ``text`` unchanged when no
        match is found.
        """
        needle = f"`{content}`"
        mut_needle = f"`{mut_content}`"
        idx = text.find(needle)
        if idx == -1:
            return text
        return text[:idx] + mut_needle + text[idx + len(needle):]

    @staticmethod
    def _split_cli_line(line: str) -> Tuple[str, List[str]]:
        """Return ``(prefix, tokens)``: prefix is leading whitespace + optional ``$ ``/``> `` prompt."""
        i = 0
        while i < len(line) and line[i] in " \t":
            i += 1
        prompt_end = i
        for p in ("$ ", "> "):
            if line.startswith(p, i):
                prompt_end = i + len(p)
                while prompt_end < len(line) and line[prompt_end] in " \t":
                    prompt_end += 1
                break
        prefix = line[:prompt_end]
        body = line[prompt_end:]
        return prefix, body.split()

    @staticmethod
    def _pick_long_flag(tokens: List[str]) -> "Tuple[int, str] | None":
        """Return ``(index, token)`` of the first long flag eligible for truncation, or None."""
        for idx, t in enumerate(tokens):
            if not t.startswith("--"):
                continue
            flag = t.split("=", 1)[0]
            if flag in _CLI_PROTECTED_FLAGS or len(flag) < 5:
                continue
            return (idx, t)
        return None

    @staticmethod
    def _permute_long_flag(token: str) -> str:
        """Transpose two adjacent characters inside a long flag's stem.

        Preserves the leading ``--`` and an optional ``=value`` suffix, e.g.
        ``--size`` → ``--szie``, ``--epochs=20`` → ``--epohcs=20``.  Returns
        the original token if the stem is too short or has no transposable
        adjacent pair.
        """
        if "=" in token:
            flag, _, value = token.partition("=")
            suffix = f"={value}"
        else:
            flag = token
            suffix = ""
        if not flag.startswith("--") or len(flag) <= 4:
            return token
        stem = flag[2:]
        mutated_stem = LLMErrorInjector._transpose_name(stem)
        if mutated_stem == stem:
            return token
        return f"--{mutated_stem}{suffix}"

    @staticmethod
    def _replace_flag_globally(text: str, orig_flag: str, mut_flag: str) -> Tuple[str, int]:
        """Replace every standalone occurrence of ``orig_flag`` with ``mut_flag``.

        Whitespace-anchored on both sides via negative lookarounds on
        ``[\\w-]`` so the match doesn't extend into longer flag names
        (``--label`` is NOT matched inside ``--label-extra`` or ``--label_size``)
        but still matches the flag when it appears bare, in usage stanzas
        ``[--flag]``, with ``=value``, or inside backticks.  Returns
        ``(new_text, n_replacements)``.
        """
        pattern = re.compile(rf"(?<![\w-]){re.escape(orig_flag)}(?![\w-])")
        new_text, n = pattern.subn(mut_flag, text)
        return new_text, n

    @staticmethod
    def _splice_bogus_flag_into_help_block(
        text: str, bogus_flag: str, bogus_val: str
    ) -> Tuple[str, int]:
        """Splice a bogus flag into any argparse-style help block in ``text``.

        Closes the cli_unknown_flag loophole where a model can look at the
        preserved ``usage:`` / ``options:`` stanza, notice the appended
        flag isn't listed, and strip it without ever consulting the repo.
        After this rewrite the doc is self-consistent in its wrongness:
        the bogus flag appears both in shell-invocation lines AND in the
        help block that claims to define valid flags.

        Idempotent — if the flag is already present in the help block
        (e.g. an earlier cli_unknown_flag entry already spliced it), this
        is a no-op for that surface.  Returns ``(new_text, n_edits)``
        where ``n_edits`` counts each surface (usage line, options entry)
        that was actually modified — ``0`` means "no help block to edit
        or flag already there".
        """
        n_edits = 0

        # Whitespace-anchored presence check so ``--workers`` does not
        # match inside ``--workers-pool`` or similar.
        already_present = re.compile(rf"(?<![\w-]){re.escape(bogus_flag)}(?![\w-])")

        # 1. Splice into the ``usage:`` block.  Append ``[--flag VAL]`` to
        # the end of the captured block — argparse's own behaviour when a
        # new option is added to a parser.  Edit only the FIRST usage
        # block; if a doc has multiple, downstream entries can target
        # them via subsequent injector calls.
        m = _HELP_USAGE_BLOCK_RE.search(text)
        if m and not already_present.search(m.group(0)):
            value_token = bogus_val.upper() if bogus_val.isalpha() else "VAL"
            new_block = m.group(0).rstrip() + f" [{bogus_flag} {value_token}]"
            text = text[: m.start()] + new_block + text[m.end():]
            n_edits += 1

        # 2. Splice into the ``options:`` body.  Append a new entry that
        # matches argparse's default formatting (2-space indent, flag +
        # value placeholder, then a 24-space description indent on the
        # next line).  Re-detect the block after the usage edit so spans
        # are accurate.
        m = _HELP_OPTIONS_BLOCK_RE.search(text)
        if m and not already_present.search(m.group("body")):
            value_token = bogus_val.upper() if bogus_val.isalpha() else "VAL"
            new_entry = (
                f"  {bogus_flag} {value_token}\n"
                f"                        Number of parallel workers.\n"
            )
            insert_at = m.end("body")
            text = text[:insert_at] + new_entry + text[insert_at:]
            n_edits += 1

        return text, n_edits

    @staticmethod
    def _pick_program_token(tokens: List[str]) -> "Tuple[int, str] | None":
        """Return ``(index, token)`` of a script-name token, or None.

        Recognises ``<launcher> script.py`` and bare ``script.py``; non-script
        first tokens like ``make`` are ignored.
        """
        if not tokens:
            return None
        first = tokens[0]
        if first in _CLI_LAUNCHERS or first == "R":
            if len(tokens) < 2:
                return None
            candidate = tokens[1]
            if candidate.startswith("-"):
                return None
            if candidate.endswith(_CLI_SCRIPT_SUFFIXES):
                return (1, candidate)
            return None
        if first.endswith(_CLI_SCRIPT_SUFFIXES):
            return (0, first)
        return None

    @staticmethod
    def _replace_token_in_line(line: str, orig_token: str, mut_token: str) -> "str | None":
        """Swap one whitespace-delimited occurrence of ``orig_token``.

        Uses negative-lookaround anchors instead of ``\\b`` so flags that start
        with ``-`` (which ``\\b`` would treat as a word boundary on the wrong
        side) match correctly.  Preserves the line's internal whitespace
        (multi-space alignment columns common in argparse output).  Returns
        the rewritten line, or ``None`` if the token wasn't found as a
        standalone occurrence.
        """
        pattern = re.compile(rf"(?<!\S){re.escape(orig_token)}(?!\S)")
        new_line, n = pattern.subn(mut_token, line, count=1)
        return new_line if n == 1 else None

    def _mutate_program_token(self, token: str) -> str:
        """Mutate the script-name stem with ``_transpose_name``; preserve path and extension."""
        if "/" in token:
            head, _, base = token.rpartition("/")
            path_prefix = head + "/"
        else:
            base = token
            path_prefix = ""
        if "." in base:
            stem, _, ext = base.rpartition(".")
            ext_suffix = "." + ext
        else:
            stem = base
            ext_suffix = ""
        if len(stem) < 4:
            return token
        mutated_stem = self._transpose_name(stem)
        if mutated_stem == stem:
            return token
        return f"{path_prefix}{mutated_stem}{ext_suffix}"

    @staticmethod
    def _record_skip(data: Dict[str, Any], category: str, reason: str) -> None:
        data.setdefault("skipped", []).append({"category": category, "reason": reason})

    def inject(self, readme_text: str, min_per_category: int = 3, preserve_keywords: list[str] | None = None, max_words: int = 450, project_terms: list[str] | None = None, force_deterministic: bool | None = None, file_type: str = "") -> Tuple[str, Dict[str, Any]]:
        # Resolve the per-call override; fall back to instance default.
        use_det = self.force_deterministic if force_deterministic is None else force_deterministic

        if use_det:
            # Skip the LLM entirely — deterministic inject + supplements only.
            corrupted, data = self._deterministic_inject(readme_text, file_type=file_type)
            corrupted, data = self._supplement_errors(
                readme_text, corrupted, data, min_per_category, project_terms, file_type=file_type
            )
            if not self._check_code_blocks_preserved(readme_text, corrupted):
                # Supplements broke fences — fall back to bare deterministic output.
                corrupted, data = self._deterministic_inject(readme_text, file_type=file_type)
            return corrupted, {
                "errors": data.get("errors", []),
                "skipped": data.get("skipped", []),
            }

        conv = CommonConversation(self.llm)
        preserve_keywords = preserve_keywords or self._extract_preserve_keywords(readme_text)
        
        # Add project terms to prompt if available
        project_terms_section = ""
        if project_terms:
            terms_str = ", ".join(project_terms[:20])  # Limit to top 20 to avoid clutter
            project_terms_section = f"\nPROJECT SPECIFIC TARGETS (Prioritize misspelling these):\n{terms_str}\n"
            
        system_prompt = escape_braces(INJECTION_PROMPT).format(
            readme=readme_text[:30000],
            min_per_category=min_per_category,
            keywords=", ".join(preserve_keywords) if preserve_keywords else "",
            max_words=max_words,
        )
        
        if project_terms:
            # Insert project terms section before ERROR CATEGORIES
            system_prompt = system_prompt.replace("ERROR CATEGORIES (inject all)", f"{project_terms_section}\nERROR CATEGORIES (inject all)")

        output, _ = conv.generate(system_prompt=system_prompt, instruction_prompt="Return the JSON now.")
        
        # Enhanced JSON parsing with better error handling
        data = self._parse_json_output(output, readme_text)
        corrupted = data.get("corrupted_markdown", readme_text)
        
        # CRITICAL: Check code block preservation before validation
        if not self._check_code_blocks_preserved(readme_text, corrupted):
            print("Warning: LLM output broke code blocks, using deterministic fallback")
            corrupted, data = self._deterministic_inject(readme_text, file_type=file_type)
        # Validate output stays within original context; fallback to deterministic if invalid
        elif not self._validate_corrupted(readme_text, corrupted, preserve_keywords):
            corrupted, data = self._deterministic_inject(readme_text, file_type=file_type)

        # Supplement to satisfy minimum per-category counts using deterministic local edits
        corrupted, data = self._supplement_errors(readme_text, corrupted, data, min_per_category, project_terms, file_type=file_type)

        # Final safety check: ensure code blocks are still intact after supplements
        if not self._check_code_blocks_preserved(readme_text, corrupted):
            print("Warning: Supplements broke code blocks, reverting to baseline with minimal errors")
            corrupted, data = self._deterministic_inject(readme_text, file_type=file_type)
        
        manifest = {
            "errors": data.get("errors", []),
            "skipped": data.get("skipped", []),
        }
        return corrupted, manifest
    
    def _check_code_blocks_preserved(self, baseline: str, corrupted: str) -> bool:
        """Check that code block structure is preserved exactly."""
        # Count code fence lines (must match exactly)
        base_fences = len(re.findall(r"^```", baseline, flags=re.M))
        corr_fences = len(re.findall(r"^```", corrupted, flags=re.M))
        if base_fences != corr_fences:
            return False
        
        # Check RMarkdown chunks specifically (```{r}, ```{python}, etc.)
        base_rmd = re.findall(r"^```\{[^}]*\}", baseline, flags=re.M)
        corr_rmd = re.findall(r"^```\{[^}]*\}", corrupted, flags=re.M)
        if len(base_rmd) != len(corr_rmd):
            return False
        
        # Ensure closing ``` match opening count
        base_close = len(re.findall(r"^```\s*$", baseline, flags=re.M))
        corr_close = len(re.findall(r"^```\s*$", corrupted, flags=re.M))
        if base_close != corr_close:
            return False
        
        return True

    def _parse_json_output(self, output: str, fallback_text: str) -> Dict[str, Any]:
        """Enhanced JSON parsing with multiple fallback strategies."""
        import re
        
        # Strategy 1: Direct JSON parsing
        try:
            return json.loads(output)
        except json.JSONDecodeError:
            pass
        
        # Strategy 2: Extract JSON block between ```json and ```
        json_pattern = r'```(?:json)?\s*(\{.*?\})\s*```'
        match = re.search(json_pattern, output, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass
        
        # Strategy 3: Find first complete JSON object
        start = output.find("{")
        if start != -1:
            # Find matching closing brace
            brace_count = 0
            end = start
            for i, char in enumerate(output[start:], start):
                if char == "{":
                    brace_count += 1
                elif char == "}":
                    brace_count -= 1
                    if brace_count == 0:
                        end = i
                        break
            
            if brace_count == 0:  # Found complete JSON object
                try:
                    json_str = output[start:end+1]
                    return json.loads(json_str)
                except json.JSONDecodeError:
                    pass
        
        # Strategy 4: Try to fix common JSON issues
        try:
            # Remove markdown code fences
            cleaned = re.sub(r'```(?:json)?\s*', '', output)
            cleaned = re.sub(r'```\s*$', '', cleaned)
            # Remove leading/trailing whitespace
            cleaned = cleaned.strip()
            # Try parsing again
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass
        
        # Strategy 5: Fallback to deterministic injection
        print(f"Warning: Failed to parse LLM JSON output, using fallback. Output preview: {output[:200]}...")
        return {"corrupted_markdown": fallback_text, "errors": []}

    def _extract_preserve_keywords(self, text: str) -> List[str]:
        # Extract capitalized terms, domain hyphenations, and hostnames in links
        kws: Set[str] = set()
        for m in re.finditer(r"\b[A-Z][A-Za-z0-9\-/]{2,}(?:\s[A-Z][A-Za-z0-9\-/]{2,})*\b", text):
            term = m.group(0)
            if len(term) <= 40:
                kws.add(term)
        for m in re.finditer(r"\b[\w]+-[\w]+\b", text):
            if any(ch.isalpha() for ch in m.group(0)):
                kws.add(m.group(0))
        for m in re.finditer(r"https?://([^/\s)]+)", text):
            kws.add(m.group(1))
        # Keep a small set to avoid over-constraining
        out = list(kws)[:20]
        return out

    def _validate_corrupted(self, baseline: str, corrupted: str, preserve_keywords: List[str]) -> bool:
        # Similarity threshold - increased for better structure preservation
        ratio = SequenceMatcher(None, baseline, corrupted).ratio()
        if ratio < 0.75:
            return False
        # Preserve keywords
        for k in preserve_keywords:
            if k and k not in corrupted:
                return False
        # No new top-level sections
        base_h2 = set([ln.strip() for ln in baseline.splitlines() if ln.strip().startswith("## ")])
        corr_h2 = set([ln.strip() for ln in corrupted.splitlines() if ln.strip().startswith("## ")])
        if not corr_h2.issubset(base_h2.union({"## Overview", "## Hardware Requirements", "## License", "## Usage", "## Dependencies", "## System Requirements"})):
            return False
        # New token ratio
        btoks = set(re.findall(r"[A-Za-z0-9_\-]+", baseline.lower()))
        ctoks = set(re.findall(r"[A-Za-z0-9_\-]+", corrupted.lower()))
        new_ratio = len(ctoks - btoks) / max(1, len(ctoks))
        if new_ratio > 0.25:
            return False
        # CRITICAL: Preserve code block structure
        # Count code fences (``` or ```{...}) - must match
        base_fences = len(re.findall(r"^```", baseline, flags=re.M))
        corr_fences = len(re.findall(r"^```", corrupted, flags=re.M))
        if base_fences != corr_fences:
            return False
        # Check RMarkdown chunks specifically
        base_rmd_chunks = len(re.findall(r"^```\{[^}]*\}", baseline, flags=re.M))
        corr_rmd_chunks = len(re.findall(r"^```\{[^}]*\}", corrupted, flags=re.M))
        if base_rmd_chunks != corr_rmd_chunks:
            return False
        return True

    def _deterministic_inject(self, baseline: str, file_type: str = "") -> Tuple[str, Dict[str, Any]]:
        errors: List[Dict[str, Any]] = []
        text = baseline
        # Pre-compute skip spans (YAML frontmatter + code fences) for prose-only mutations
        _det_yaml = self._yaml_frontmatter_span(text)
        _det_fences = self._fence_spans(text)
        _det_skip = sorted(_det_fences + ([_det_yaml] if _det_yaml else []))

        def _det_prose_replace(t: str, old: str, new: str) -> str:
            return self._replace_prose_only(t, old, new, _det_skip)

        # typo
        if "successfully" in text:
            result = _det_prose_replace(text, "successfully", "succesfully")
            if result != text:
                text = result
                errors.append({"id": "e_typo_1", "category": "typo", "original_snippet": "successfully", "mutated_snippet": "succesfully", "rationale": "common misspelling"})
        elif "installation" in text:
            result = _det_prose_replace(text, "installation", "instalation")
            if result != text:
                text = result
                errors.append({"id": "e_typo_1", "category": "typo", "original_snippet": "installation", "mutated_snippet": "instalation", "rationale": "common misspelling"})
        # link — skip if match falls inside YAML or a code fence
        m = next(
            (m for m in re.finditer(r"\]\(https?://[^)]+\)", text)
             if not any(s <= m.start() < e for s, e in _det_skip)),
            None,
        )
        if m:
            broken = m.group(0).replace("https://", "https//")
            text = text[:m.start()] + broken + text[m.end():]
            errors.append({"id": "e_link_1", "category": "link", "original_snippet": m.group(0), "mutated_snippet": broken, "rationale": "missing colon in scheme"})
        # duplicate a small section (next header and paragraph)
        lines = text.splitlines()
        dup_idx = next((i for i, ln in enumerate(lines) if ln.strip().startswith("## ")), None)
        if dup_idx is not None:
            block = lines[dup_idx: min(len(lines), dup_idx+5)]
            text = "\n".join(lines + ["", *block])
            errors.append({"id": "e_dup_1", "category": "duplicate", "original_snippet": "\n".join(block), "mutated_snippet": "\n".join(block), "rationale": "duplicated section"})
        # markdown structure: break a header (skip YAML and code fences)
        _det_fences = self._fence_spans(text)
        _det_yaml = self._yaml_frontmatter_span(text)
        _det_skip = sorted(_det_fences + ([_det_yaml] if _det_yaml else []))
        _fence_sp = _det_skip  # alias for clarity below
        _md_match = next(
            (m for m in re.finditer(r"\n# ", text)
             if not any(s <= m.start() < e for s, e in _fence_sp)),
            None,
        )
        if _md_match:
            pos = _md_match.start()
            text = text[:pos] + "\n#" + text[pos + len("\n# "):]
            errors.append({"id": "e_md_1", "category": "markdown_structure", "original_snippet": "\n# ", "mutated_snippet": "\n#", "rationale": "missing space in header"})
        data: Dict[str, Any] = {"errors": errors}

        # accession_id_prefix: swap GSE <-> GSM only when a context word (series/samples/...) sits near it
        prose = self._prose_region(text)
        acc_match = None
        for m in re.finditer(r"\b(GSE|GSM)(\d{3,})\b", prose):
            ctx_start = max(0, m.start() - 80)
            ctx_end = min(len(prose), m.end() + 80)
            if any(re.search(rf"\b{w}\b", prose[ctx_start:ctx_end], re.I) for w in _ACCESSION_CONTEXT_WORDS):
                acc_match = m
                break
        if acc_match:
            prefix = acc_match.group(1)
            digits = acc_match.group(2)
            orig_acc = prefix + digits
            mut_acc = ("GSM" if prefix == "GSE" else "GSE") + digits
            text = text.replace(orig_acc, mut_acc, 1)
            errors.append({"id": "e_accession_1", "category": "accession_id_prefix", "original_snippet": orig_acc, "mutated_snippet": mut_acc, "rationale": "swapped accession namespace against surrounding context word"})
        else:
            self._record_skip(data, "accession_id_prefix", "no_context_anchor")

        # prose_code_pkg_version: prose version disagrees with version pinned/loaded in a code fence
        code = self._extract_code_fragments(text)
        pkg_anchor = _PKG_ANCHOR_RE.search(code)
        if pkg_anchor:
            pkg_name = pkg_anchor.group(1)
            code_ver = int(pkg_anchor.group(2))
            prose_view = self._prose_region(text)
            mp = re.search(rf"\b({re.escape(pkg_name)})\s+v?(\d+)(\.\d+)?\b", prose_view, re.I)
            if mp and int(mp.group(2)) == code_ver:
                new_v = code_ver - 1 if code_ver > 1 else code_ver + 1
                orig_pv = mp.group(0)
                mut_pv = f"{mp.group(1)} v{new_v}"
                text = text.replace(orig_pv, mut_pv, 1)
                errors.append({"id": "e_pkg_ver_1", "category": "prose_code_pkg_version", "original_snippet": orig_pv, "mutated_snippet": mut_pv, "rationale": f"prose version drifted from code-pinned {pkg_name} v{code_ver}"})
            else:
                self._record_skip(data, "prose_code_pkg_version", "no_prose_version_match")
        else:
            self._record_skip(data, "prose_code_pkg_version", "no_anchor")

        # prose_code_stat_test: prose test name disagrees with fn called in a code fence
        code = self._extract_code_fragments(text)
        stat_anchor = _STAT_TEST_ANCHOR_RE.search(code)
        if stat_anchor:
            hit = stat_anchor.group(0).lower()
            # Seurat's FindMarkers / FindAllMarkers default to Wilcoxon when
            # test.use is not specified, so treat them as wilcoxon anchors.
            code_test = "wilcoxon" if ("wilcox" in hit or "findmarkers" in hit or "findallmarkers" in hit) else "t-test"
            prose_view = self._prose_region(text)
            if code_test == "wilcoxon":
                mp = re.search(r"\b(t[- ]test|Student'?s t|two-sample t)\b", prose_view, re.I)
            else:
                mp = re.search(r"\b(Wilcoxon(?: rank[- ]sum)?|Mann[- ]Whitney)\b", prose_view, re.I)
            if mp:
                orig_st = mp.group(0)
                mut_st = "t-test" if code_test == "wilcoxon" else "Wilcoxon"
                text = text.replace(orig_st, mut_st, 1)
                errors.append({"id": "e_stat_1", "category": "prose_code_stat_test", "original_snippet": orig_st, "mutated_snippet": mut_st, "rationale": f"prose test name contradicts code-called {code_test}"})
            else:
                self._record_skip(data, "prose_code_stat_test", "no_prose_test_match")
        else:
            self._record_skip(data, "prose_code_stat_test", "no_anchor")

        # prose_code_marker: prose marker disagrees with marker referenced in a code fence
        code = self._extract_code_fragments(text)
        marker_hit = _MARKER_ANCHOR_RE.search(code)
        if marker_hit:
            # Group 1: features=c("X"); Group 2: subset(..., X >); Group 3: FeaturePlot/VlnPlot/...
            code_marker = marker_hit.group(1) or marker_hit.group(2) or marker_hit.group(3)
            swap_map = {"CD4": "CD8", "CD8": "CD4", "FOXP3": "RORC", "GATA3": "TBX21"}
            if code_marker in swap_map:
                prose_view = self._prose_region(text)
                mp = re.search(rf"\b{re.escape(code_marker)}\b", prose_view)
                if mp:
                    orig_mk = mp.group(0)
                    mut_mk = swap_map[code_marker]
                    text = text.replace(orig_mk, mut_mk, 1)
                    errors.append({"id": "e_marker_1", "category": "prose_code_marker", "original_snippet": orig_mk, "mutated_snippet": mut_mk, "rationale": f"prose marker contradicts code-referenced {code_marker}"})
                else:
                    self._record_skip(data, "prose_code_marker", "no_prose_marker_match")
            else:
                self._record_skip(data, "prose_code_marker", "unmapped_code_marker")
        else:
            self._record_skip(data, "prose_code_marker", "no_anchor")

        # prose_code_param: prose hyperparameter disagrees with value in a code-fence function call
        code = self._extract_code_fragments(text)
        param_hit = _PARAM_ANCHOR_RE.search(code)
        if param_hit:
            param_name = param_hit.group(1)
            code_val = param_hit.group(2)
            prose_view = self._prose_region(text)
            mp = re.search(rf"{re.escape(param_name)}\s+(?:of\s+|=\s*)?({re.escape(code_val)})\b", prose_view, re.I)
            if mp:
                try:
                    num = float(code_val)
                    new_val = f"{num + 0.1:.1f}" if "." in code_val else str(int(num) + 1)
                except ValueError:
                    new_val = code_val + "x"
                orig_pm = mp.group(0)
                mut_pm = orig_pm.replace(code_val, new_val, 1)
                text = text.replace(orig_pm, mut_pm, 1)
                errors.append({"id": "e_param_1", "category": "prose_code_param", "original_snippet": orig_pm, "mutated_snippet": mut_pm, "rationale": f"prose {param_name} value drifted from code-used {code_val}"})
            else:
                self._record_skip(data, "prose_code_param", "no_prose_value_match")
        else:
            self._record_skip(data, "prose_code_param", "no_anchor")

        # code consistency: comment conflicts inside fences
        text, errors = self._inject_code_consistency(text, errors, data, file_type=file_type)

        # CLI consistency: flag typos / bogus flags / script renames inside fences
        text, errors = self._inject_cli_consistency(text, errors, data, file_type=file_type)

        # .Rd prose reference errors (function/arg names in documentation prose)
        if self._is_rd_file(text):
            text, errors = self._inject_rd_errors(text, errors, data)

        data["errors"] = errors

        return text, data

    def _inject_code_consistency(
        self,
        text: str,
        errors: List[Dict[str, Any]],
        data: Dict[str, Any],
        file_type: str = "",
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """Inject code-consistency errors, gated by file_type.

        - .rmd / .ipynb: skip all categories (executable code, not doc-quality scope)
        - .md / .rst / .txt: inject code_func_name, code_func_args, code_comment_conflict
        - other/unknown: inject only code_comment_conflict
        """
        ft = file_type.lower()
        fence_spans = self._fence_spans(text)
        if not fence_spans:
            self._record_skip(data, "code_comment_conflict", "no_code_block")
            if ft in _PROSE_CODE_EXTENSIONS:
                self._record_skip(data, "code_func_name", "no_code_block")
                self._record_skip(data, "code_func_args", "no_code_block")
            elif ft not in _EXECUTABLE_CODE_EXTENSIONS:
                self._record_skip(data, "code_func_name", "unknown_file_type")
                self._record_skip(data, "code_func_args", "unknown_file_type")
            return text, errors

        if ft in _EXECUTABLE_CODE_EXTENSIONS:
            self._record_skip(data, "code_comment_conflict", "executable_code_file")
            self._record_skip(data, "code_func_name", "executable_code_file")
            self._record_skip(data, "code_func_args", "executable_code_file")
            return text, errors

        used: set = {e.get("original_snippet", "") for e in errors}

        # ── code_func_name (prose files only) ─────────────────────────────────
        if ft in _PROSE_CODE_EXTENSIONS:
            code_view = self._extract_code_fragments(text)
            injected_fn = False
            for m in _FUNC_IN_CODE_RE.finditer(code_view):
                name = m.group(1)
                if name.lower() in _CODE_SKIP_NAMES or len(name) < 5:
                    continue
                orig_call = m.group(0)
                if orig_call in used:
                    continue
                mutated = self._transpose_name(name)
                if mutated == name:
                    continue
                mut_call = mutated + orig_call[len(name):]
                new_text = self._replace_in_fence(text, orig_call, mut_call, fence_spans)
                if new_text != text:
                    errors.append({
                        "id": f"e_cfn_{len(errors)}", "category": "code_func_name",
                        "original_snippet": orig_call.rstrip(), "mutated_snippet": mut_call.rstrip(),
                        "rationale": "transposed characters in function name inside code fence",
                    })
                    used.add(orig_call)
                    text = new_text
                    fence_spans = self._fence_spans(text)
                    injected_fn = True
                    break
            if not injected_fn:
                self._record_skip(data, "code_func_name", "no_suitable_function_call")

            # ── code_func_args ─────────────────────────────────────────────────
            code_view = self._extract_code_fragments(text)
            injected_fa = False
            for m in _NAMED_ARG_IN_CODE_RE.finditer(code_view):
                name = m.group(1)
                if name.lower() in _CODE_SKIP_NAMES or len(name) < 3:
                    continue
                orig_arg = m.group(0)
                if orig_arg in used:
                    continue
                mutated = self._transpose_name(name)
                if mutated == name:
                    continue
                mut_arg = mutated + orig_arg[len(name):]
                new_text = self._replace_in_fence(text, orig_arg, mut_arg, fence_spans)
                if new_text != text:
                    errors.append({
                        "id": f"e_cfa_{len(errors)}", "category": "code_func_args",
                        "original_snippet": orig_arg.rstrip(), "mutated_snippet": mut_arg.rstrip(),
                        "rationale": "transposed characters in named argument inside code fence",
                    })
                    used.add(orig_arg)
                    text = new_text
                    fence_spans = self._fence_spans(text)
                    injected_fa = True
                    break
            if not injected_fa:
                self._record_skip(data, "code_func_args", "no_suitable_named_arg")
        else:
            self._record_skip(data, "code_func_name", "unknown_file_type")
            self._record_skip(data, "code_func_args", "unknown_file_type")

        # ── code_comment_conflict ─────────────────────────────────────────────
        injected = False
        code_view = self._extract_code_fragments(text)
        for m in _COMMENT_IN_CODE_RE.finditer(code_view):
            prefix = m.group(1)
            comment_body = m.group(2)
            for pat, replacement in _COMMENT_CONFLICT_MAP:
                if not pat.search(comment_body):
                    continue
                mutated_body = pat.sub(replacement, comment_body, count=1)
                if mutated_body == comment_body:
                    continue
                orig_line = prefix + comment_body
                mut_line = prefix + mutated_body
                if orig_line in used or orig_line.rstrip() in used:
                    continue
                new_text = self._replace_in_fence(text, orig_line, mut_line, fence_spans)
                if new_text != text:
                    errors.append({
                        "id": f"e_ccc_{len(errors)}", "category": "code_comment_conflict",
                        "original_snippet": orig_line.rstrip(), "mutated_snippet": mut_line.rstrip(),
                        "rationale": "comment changed to conflict with adjacent code",
                    })
                    used.add(orig_line)
                    text = new_text
                    fence_spans = self._fence_spans(text)
                    injected = True
                    break
            if injected:
                break
        if not injected:
            self._record_skip(data, "code_comment_conflict", "no_suitable_comment")

        return text, errors

    def _inject_cli_consistency(
        self,
        text: str,
        errors: List[Dict[str, Any]],
        data: Dict[str, Any],
        file_type: str = "",
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """Inject CLI-consistency errors into shell-style invocation lines.

        Three categories:
          * ``cli_flag_typo`` — truncate a long ``--flag`` so the parser no
            longer recognises it (``--epochs`` → ``--epoch``).
          * ``cli_unknown_flag`` — append a bogus flag the parser does not
            define (``--workers 4`` round-robin).
          * ``cli_program_rename`` — mutate the script-name stem so the file
            does not exist in the repo (``train.py`` → ``trian.py``).

        Gating mirrors ``_inject_code_consistency``: only ``.md``/``.rst``/
        ``.txt`` are mutated; ``.rmd``/``.ipynb`` and unknown types are
        skipped.  Each category fires at most once per call.
        """
        cli_cats = ("cli_flag_typo", "cli_unknown_flag", "cli_program_rename")
        ft = file_type.lower()
        if ft in _EXECUTABLE_CODE_EXTENSIONS:
            for c in cli_cats:
                self._record_skip(data, c, "executable_code_file")
            return text, errors
        if ft not in _PROSE_CODE_EXTENSIONS:
            for c in cli_cats:
                self._record_skip(data, c, "unknown_file_type")
            return text, errors

        fence_spans = self._fence_spans(text)
        # NOTE: we don't bail out on "no fences" because inline-code CLI
        # invocations (e.g. pharokka-style `` `script.py --flag x` ``) are
        # valid candidates even when the doc has no triple-backtick blocks.

        used_snippets: Set[str] = {e.get("original_snippet", "") for e in errors}
        used_lines: Set[str] = set()
        candidates = self._cli_candidates(text)
        if not candidates:
            for c in cli_cats:
                self._record_skip(data, c, "no_cli_invocation_line")
            return text, errors

        def _apply(line: str, mut_line: str, source: str, spans):
            """Apply substitution scoped to where the candidate came from."""
            if source == "fence":
                return self._replace_in_fence(text, line, mut_line, spans)
            return self._replace_in_inline_code(text, line, mut_line)

        # ── cli_flag_typo ────────────────────────────────────────────────
        # Pick a long flag from one real CLI invocation line, then permute its
        # stem and rewrite EVERY occurrence of that flag throughout the doc —
        # fenced blocks, inline-code spans, usage stanzas, prose mentions —
        # so the document stays internally consistent in its wrongness.
        injected = False
        for line, source in candidates:
            if line in used_lines or line in used_snippets:
                continue
            _prefix, tokens = self._split_cli_line(line)
            picked = self._pick_long_flag(tokens)
            if not picked:
                continue
            _idx, orig_token = picked
            mut_token = self._permute_long_flag(orig_token)
            if mut_token == orig_token:
                continue
            # Strip any "=value" suffix — we replace just the flag stem so
            # both ``--foo bar`` and ``--foo=bar`` forms get caught.
            orig_flag = orig_token.split("=", 1)[0]
            mut_flag = mut_token.split("=", 1)[0]
            new_text, n_replaced = self._replace_flag_globally(text, orig_flag, mut_flag)
            if n_replaced == 0 or new_text == text:
                continue
            # mutated_snippet records the picked-from line after the global
            # rewrite — the canonical worked example for diagnostics.
            mut_line = self._replace_token_in_line(line, orig_token, mut_token) or line
            errors.append({
                "id": f"e_cli_flag_typo_{len(errors)}",
                "category": "cli_flag_typo",
                "original_snippet": line,
                "mutated_snippet": mut_line,
                "original_token": orig_flag,
                "mutated_token": mut_flag,
                "occurrences_changed": n_replaced,
                "rationale": f"permuted long flag {orig_flag} -> {mut_flag} ({n_replaced} occurrences)",
            })
            used_lines.add(line)
            used_lines.add(mut_line)
            text = new_text
            fence_spans = self._fence_spans(text)
            candidates = self._cli_candidates(text)
            injected = True
            break
        if not injected:
            self._record_skip(data, "cli_flag_typo", "no_long_flag")

        # ── cli_unknown_flag ─────────────────────────────────────────────
        # After appending the bogus flag to a shell invocation, also splice
        # it into any argparse-style help block so the doc is internally
        # self-consistent in its wrongness.  Without this, models can
        # trivially cross-reference the preserved help block — see
        # ``_splice_bogus_flag_into_help_block`` for the rationale.
        injected = False
        bogus_flag, bogus_val = _CLI_BOGUS_FLAGS[len(errors) % len(_CLI_BOGUS_FLAGS)]
        for line, source in candidates:
            if line in used_lines or line in used_snippets:
                continue
            _prefix, tokens = self._split_cli_line(line)
            if not tokens or bogus_flag in tokens:
                continue
            mut_line = line.rstrip() + f" {bogus_flag} {bogus_val}"
            new_text = _apply(line, mut_line, source, fence_spans)
            if new_text != text:
                new_text, n_help_edits = self._splice_bogus_flag_into_help_block(
                    new_text, bogus_flag, bogus_val
                )
                rationale = f"appended bogus flag {bogus_flag}"
                if n_help_edits:
                    rationale += f" + spliced into help block ({n_help_edits} surface)"
                errors.append({
                    "id": f"e_cli_unknown_flag_{len(errors)}",
                    "category": "cli_unknown_flag",
                    "original_snippet": line,
                    "mutated_snippet": mut_line,
                    "mutated_token": bogus_flag,
                    "help_block_edits": n_help_edits,
                    "rationale": rationale,
                })
                used_lines.add(line)
                used_lines.add(mut_line)
                text = new_text
                fence_spans = self._fence_spans(text)
                candidates = self._cli_candidates(text)
                injected = True
                break
        if not injected:
            self._record_skip(data, "cli_unknown_flag", "no_cli_invocation_line")

        # ── cli_program_rename ───────────────────────────────────────────
        injected = False
        for line, source in candidates:
            if line in used_lines or line in used_snippets:
                continue
            _prefix, tokens = self._split_cli_line(line)
            picked = self._pick_program_token(tokens)
            if not picked:
                continue
            _idx, orig_token = picked
            mut_token = self._mutate_program_token(orig_token)
            if mut_token == orig_token:
                continue
            mut_line = self._replace_token_in_line(line, orig_token, mut_token)
            if mut_line is None:
                continue
            new_text = _apply(line, mut_line, source, fence_spans)
            if new_text != text:
                errors.append({
                    "id": f"e_cli_program_rename_{len(errors)}",
                    "category": "cli_program_rename",
                    "original_snippet": line,
                    "mutated_snippet": mut_line,
                    "mutated_token": mut_token,
                    "rationale": f"mutated program {orig_token} -> {mut_token}",
                })
                used_lines.add(line)
                used_lines.add(mut_line)
                text = new_text
                fence_spans = self._fence_spans(text)
                injected = True
                break
        if not injected:
            self._record_skip(data, "cli_program_rename", "no_script_token")

        return text, errors

    def _inject_rd_errors(
        self,
        text: str,
        errors: List[Dict[str, Any]],
        data: Dict[str, Any],
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """Inject rd_func_name and rd_arg_name errors into .Rd prose sections.

        Only mutates text outside \\usage{} and \\examples{} blocks.
        - rd_func_name: transpose two chars in a \\code{FuncName} or \\link{FuncName}
          reference in prose.
        - rd_arg_name: transpose two chars in an argument name mentioned in
          \\item{arg}{description} prose (either as \\code{arg} or bare word).
        """
        skip_spans = self._rd_skip_spans(text)
        used: set = {e.get("original_snippet", "") for e in errors}

        # ── rd_func_name ──────────────────────────────────────────────────────
        injected = False
        for pattern in (_RD_CODE_FUNC_RE, _RD_LINK_RE):
            if injected:
                break
            for m in pattern.finditer(text):
                if self._in_fence(m.start(), skip_spans):
                    continue
                name = m.group(1)
                if name in used or len(name) < 4:
                    continue
                mutated = self._transpose_name(name)
                if mutated == name:
                    continue
                # Replace ONLY the captured group (group 1) inside the match,
                # not the first occurrence of `name` in the full snippet
                # (which might be in the package prefix of \link[pkg]{name}).
                orig_snippet = m.group(0)
                g1_start = m.start(1) - m.start()
                g1_end   = m.end(1)   - m.start()
                mut_snippet = orig_snippet[:g1_start] + mutated + orig_snippet[g1_end:]
                if mut_snippet in used:
                    continue
                new_text = self._replace_in_rd_prose(text, orig_snippet, mut_snippet, skip_spans)
                if new_text != text:
                    errors.append({
                        "id": f"e_rdfn_{len(errors)}",
                        "category": "rd_func_name",
                        "original_snippet": orig_snippet,
                        "mutated_snippet": mut_snippet,
                        "rationale": "character swap in function name in .Rd prose",
                    })
                    used.add(orig_snippet)
                    text = new_text
                    injected = True
                    break
        if not injected:
            self._record_skip(data, "rd_func_name", "no_suitable_func_reference")

        # ── rd_arg_name ───────────────────────────────────────────────────────
        # Strategy 1: \code{argname} in prose (no parens = likely argument ref)
        # Strategy 2: \item{argname}{desc} where argname appears verbatim in desc
        injected = False
        # Strategy 1
        for m in _RD_CODE_ARG_RE.finditer(text):
            if self._in_fence(m.start(), skip_spans):
                continue
            name = m.group(1)
            if name in used or name.lower() in _CODE_SKIP_NAMES or len(name) < 3:
                continue
            mutated = self._transpose_name(name)
            if mutated == name:
                continue
            orig_snippet = m.group(0)
            g1_start = m.start(1) - m.start()
            g1_end   = m.end(1)   - m.start()
            mut_snippet = orig_snippet[:g1_start] + mutated + orig_snippet[g1_end:]
            if mut_snippet in used:
                continue
            new_text = self._replace_in_rd_prose(text, orig_snippet, mut_snippet, skip_spans)
            if new_text != text:
                errors.append({
                    "id": f"e_rdan_{len(errors)}",
                    "category": "rd_arg_name",
                    "original_snippet": orig_snippet,
                    "mutated_snippet": mut_snippet,
                    "rationale": "character swap in argument name in .Rd prose",
                })
                used.add(orig_snippet)
                text = new_text
                injected = True
                break
        # Strategy 2: bare argname mention inside \item{arg}{desc}
        if not injected:
            for m in re.finditer(r"\\item\{([a-z][a-z0-9._]{2,})\}\{([^}]+)\}", text):
                if self._in_fence(m.start(), skip_spans):
                    continue
                argname = m.group(1)
                desc = m.group(2)
                # Look for a word-boundary occurrence of argname inside the description
                word_m = re.search(rf"\b{re.escape(argname)}\b", desc)
                if not word_m:
                    continue
                if argname in used or argname.lower() in _CODE_SKIP_NAMES or len(argname) < 3:
                    continue
                mutated = self._transpose_name(argname)
                if mutated == argname:
                    continue
                # Build orig/mut snippets as just the argname occurrence in desc
                desc_offset = m.start(2)  # absolute position of desc start
                abs_word_start = desc_offset + word_m.start()
                orig_snippet = argname
                mut_snippet  = mutated
                if orig_snippet in used:
                    continue
                # Replace that specific occurrence in the full text
                new_text = text[:abs_word_start] + mutated + text[abs_word_start + len(argname):]
                if new_text != text:
                    errors.append({
                        "id": f"e_rdan_{len(errors)}",
                        "category": "rd_arg_name",
                        "original_snippet": orig_snippet,
                        "mutated_snippet": mut_snippet,
                        "rationale": f"character swap in argument name '{argname}' in \\item description",
                    })
                    used.add(orig_snippet)
                    text = new_text
                    injected = True
                    break
        if not injected:
            self._record_skip(data, "rd_arg_name", "no_suitable_arg_reference")

        return text, errors

    def _supplement_errors(self, baseline: str, corrupted: str, data: Dict[str, Any], min_per_category: int, project_terms: list[str] | None = None, file_type: str = "") -> Tuple[str, Dict[str, Any]]:
        errors: List[Dict[str, Any]] = data.get("errors", []) or []
        cat_counts: Dict[str, int] = {}
        for e in errors:
            cat = e.get("category", "")
            cat_counts[cat] = cat_counts.get(cat, 0) + 1

        # Track what's already been corrupted to avoid re-corruption
        corrupted_snippets: Set[str] = set()
        for e in errors:
            corrupted_snippets.add(e.get("original_snippet", ""))
            corrupted_snippets.add(e.get("mutated_snippet", ""))

        # Pre-compute fence spans and YAML span for prose-only replacement.
        # Recomputed after each replacement to stay in sync with offsets.
        fence_spans = self._fence_spans(corrupted)
        yaml_span = self._yaml_frontmatter_span(corrupted)

        def need(cat: str) -> int:
            return max(0, min_per_category - cat_counts.get(cat, 0))

        def prose_replace(text: str, old: str, new: str) -> str:
            """Replace first occurrence of ``old`` outside fenced code blocks and YAML frontmatter."""
            nonlocal fence_spans, yaml_span
            skip = sorted(fence_spans + ([yaml_span] if yaml_span else []))
            result = self._replace_prose_only(text, old, new, skip)
            if result != text:
                fence_spans = self._fence_spans(result)
                yaml_span = self._yaml_frontmatter_span(result)
            return result

        def fence_replace(text: str, old: str, new: str) -> str:
            """Replace first occurrence of ``old`` inside fenced code blocks."""
            nonlocal fence_spans
            result = self._replace_in_fence(text, old, new, fence_spans)
            if result != text:
                fence_spans = self._fence_spans(result)
            return result

        def cli_replace(text: str, old: str, new: str, source: str) -> str:
            """Replace ``old`` with ``new`` scoped to the CLI candidate's source.

            ``source`` is ``"fence"`` for triple-backtick fence content or
            ``"inline"`` for single-backtick inline-code spans.
            """
            if source == "fence":
                return fence_replace(text, old, new)
            return self._replace_in_inline_code(text, old, new)

        def add_error(
            cat: str, orig: str, mut: str, rationale: str,
            mutated_token: Optional[str] = None,
        ) -> bool:
            """Add error and update tracking. Returns True if added.

            ``mutated_token`` is the specific substring introduced or changed
            by the mutation (e.g. ``"--cores"`` for cli_unknown_flag). When
            present it lets the evaluator do a token-anchored fix check,
            which is robust to incidental whitespace/quoting changes the LLM
            may make elsewhere on the same line.
            """
            if orig in corrupted_snippets or mut in corrupted_snippets:
                return False  # Already corrupted
            entry: Dict[str, Any] = {
                "id": f"e_{cat}_sup_{len(errors)}",
                "category": cat,
                "original_snippet": orig,
                "mutated_snippet": mut,
                "rationale": rationale,
            }
            if mutated_token is not None:
                entry["mutated_token"] = mutated_token
            errors.append(entry)
            cat_counts[cat] = cat_counts.get(cat, 0) + 1
            corrupted_snippets.add(orig)
            corrupted_snippets.add(mut)
            return True

        # Typo mutation functions for variety
        def mutate_truncate(word: str) -> str:
            """Remove last character."""
            return word[:-1] if len(word) > 3 else word + "x"
        
        def mutate_swap(word: str) -> str:
            """Swap two adjacent characters."""
            if len(word) < 4:
                return word + "e"
            pos = len(word) // 2
            return word[:pos] + word[pos+1] + word[pos] + word[pos+2:]
        
        def mutate_delete(word: str) -> str:
            """Delete a middle character."""
            if len(word) < 5:
                return word[:-1]
            pos = len(word) // 2
            return word[:pos] + word[pos+1:]
        
        def mutate_double(word: str) -> str:
            """Double a character."""
            if len(word) < 3:
                return word + word[-1]
            pos = len(word) // 2
            return word[:pos] + word[pos] + word[pos:]
        
        def mutate_case(word: str) -> str:
            """Change case of first letter."""
            if word[0].isupper():
                return word[0].lower() + word[1:]
            return word[0].upper() + word[1:]
        
        typo_mutations = [mutate_truncate, mutate_swap, mutate_delete, mutate_double]
        typo_mutation_idx = 0
        
        # typo supplements - find words to corrupt with varied mutations
        typo_attempts = 0
        max_typo_attempts = min_per_category * 5  # More attempts for variety
        
        # Priority words for typos
        priority_words = [
            "installation", "successfully", "analysis", "documentation", "maintained",
            "example", "requirements", "license", "tutorials", "expression",
            "differential", "features", "cluster", "cells", "data", "sample",
            "marker", "gene", "function", "package", "method", "parameter",
            "variable", "object", "default", "optional", "required", "specify",
            "available", "different", "following", "particular", "similar",
            "significant", "corresponding", "additional", "individual"
        ]
        
        while need("typo") > 0 and typo_attempts < max_typo_attempts:
            typo_attempts += 1
            found = False
            
            # Try priority words first
            for word in priority_words:
                pattern = r"\b" + re.escape(word) + r"\b"
                for m in re.finditer(pattern, corrupted, flags=re.I):
                    orig = m.group(0)
                    if orig in corrupted_snippets:
                        continue
                    
                    # Try different mutations
                    mutation_fn = typo_mutations[typo_mutation_idx % len(typo_mutations)]
                    typo_mutation_idx += 1
                    mut = mutation_fn(orig)
                    
                    if mut == orig or mut in corrupted_snippets:
                        continue
                    if orig not in baseline:
                        continue
                    
                    corrupted = prose_replace(corrupted, orig, mut)
                    rationale = f"{mutation_fn.__doc__.strip().lower()}"
                    if add_error("typo", orig, mut, rationale):
                        found = True
                        break
                if found:
                    break
            
            if not found:
                # Try generic words with 5+ chars
                for m in re.finditer(r"\b[A-Za-z]{5,}\b", corrupted):
                    orig = m.group(0)
                    if orig in corrupted_snippets or orig not in baseline:
                        continue
                    if orig.lower() in ["false", "true", "null", "none"]:
                        continue
                    
                    mutation_fn = typo_mutations[typo_mutation_idx % len(typo_mutations)]
                    typo_mutation_idx += 1
                    mut = mutation_fn(orig)
                    
                    if mut == orig or mut in corrupted_snippets:
                        continue
                    
                    corrupted = prose_replace(corrupted, orig, mut)
                    if add_error("typo", orig, mut, mutation_fn.__doc__.strip().lower()):
                        found = True
                        break
            
            if not found:
                break

        # link supplements - find unique links to corrupt
        link_attempts = 0
        while need("link") > 0 and link_attempts < min_per_category * 2:
            link_attempts += 1
            found = False
            for m in re.finditer(r"\[[^\]]+\]\(https?://[^)]+\)", corrupted):
                orig = m.group(0)
                if orig in corrupted_snippets:
                    continue
                mut = orig.replace("https://", "https//", 1)
                if mut == orig:
                    mut = orig.replace("http://", "http//", 1)
                if mut == orig or mut in corrupted_snippets:
                    continue
                corrupted = prose_replace(corrupted, orig, mut)
                if add_error("link", orig, mut, "scheme colon removed"):
                    found = True
                    break
            if not found:
                break

        # duplicate supplements (cap to min_per_category) - limited to avoid excessive duplication
        dup_count = 0
        max_dups = min(need("duplicate"), 5)  # Cap duplicates at 5 max
        while dup_count < max_dups:
            lines = corrupted.splitlines()
            idx = next((i for i, ln in enumerate(lines) if ln.strip().startswith("- ") or ln.strip().startswith("## ")), None)
            if idx is None:
                break
            frag = lines[idx]
            if frag in corrupted_snippets:
                break  # Already duplicated this line
            lines = lines[:idx+1] + [frag] + lines[idx+1:]
            corrupted = "\n".join(lines)
            if add_error("duplicate", frag, frag, "line duplicated"):
                dup_count += 1
            else:
                break

        # bio_term supplements
        bio_swaps = [(r"single cell", "single sell"), (r"genomics", "genomis"), (r"spatial", "spacial"),
                     (r"transcriptome", "transcriptom"), (r"proteome", "proteom"), (r"methylation", "metylation")]
        for pat, rep in bio_swaps:
            if need("bio_term") <= 0:
                break
            m = re.search(pat, corrupted, flags=re.I)
            if m:
                orig = m.group(0)
                if orig in corrupted_snippets or orig not in baseline:
                    continue
                mut = rep if orig.islower() else rep.title()
                if mut in corrupted_snippets:
                    continue
                corrupted = prose_replace(corrupted, orig, mut)
                add_error("bio_term", orig, mut, "common domain typo")

        # function supplements
        # First try project terms if available
        if project_terms:
            # Check if any existing function error targets a project term
            has_project_error = any(
                e.get("category") == "function" and 
                any(term in e.get("original_snippet", "") for term in project_terms)
                for e in errors
            )
            
            # If no project error yet, force at least one if possible
            force_project = not has_project_error
            
            for term in project_terms:
                if need("function") <= 0 and not force_project:
                    break
                
                # Look for term followed by optional parens
                m = re.search(r"\b" + re.escape(term) + r"(?:\(\)?)?", corrupted)
                if m:
                    orig = m.group(0)
                    # Skip if already corrupted
                    if orig in corrupted_snippets or orig not in baseline:
                        continue
                    
                    # Simple mutation: drop last char or append 'x'
                    if len(term) > 3:
                        mut_term = term[:-1]
                    else:
                        mut_term = term + "x"
                    
                    mut = orig.replace(term, mut_term)
                    if mut in corrupted_snippets:
                        continue
                    
                    corrupted = prose_replace(corrupted, orig, mut)
                    if add_error("function", orig, mut, f"misspelled project function {term}"):
                        if force_project:
                            force_project = False

        # Fallback to generic function detection - find unique functions
        func_attempts = 0
        while need("function") > 0 and func_attempts < min_per_category * 2:
            func_attempts += 1
            found = False
            for m in re.finditer(r"\b([A-Za-z_][A-Za-z0-9_]*)\(", corrupted):
                fname = m.group(1)
                orig = fname + "("
                
                # Skip if already corrupted or not in baseline
                if orig in corrupted_snippets or orig not in baseline:
                    continue
                # Skip project terms (handled above)
                if project_terms and fname in project_terms:
                    continue

                if len(fname) > 3:
                    mut_name = fname[:-1]
                else:
                    mut_name = fname + "x"
                mutated = mut_name + "("
                
                if mutated in corrupted_snippets:
                    continue
                
                corrupted = prose_replace(corrupted, orig, mutated)
                if add_error("function", orig, mutated, "misspelled API name"):
                    found = True
                    break
            if not found:
                break

        # markdown_structure supplements
        # NOTE: We do NOT break code fences as this destroys document structure
        # Only apply safe structural changes like header spacing
        for _ in range(need("markdown_structure")):
            # Try header space removal first (safe)
            m = re.search(r"^(#{1,6}) +", corrupted, flags=re.M)
            if m:
                orig = m.group(0)
                # Remove one space after # symbols
                mut = orig.rstrip() 
                if mut != orig:
                    corrupted = prose_replace(corrupted, orig, mut)
                    errors.append({"id": f"e_md_sup_{len(errors)}", "category": "markdown_structure", "original_snippet": orig.strip(), "mutated_snippet": mut.strip(), "rationale": "removed header space"})
                    continue
            # Try list indentation issues (safe)
            m = re.search(r"^( {2,4})[-*]", corrupted, flags=re.M)
            if m:
                orig = m.group(0)
                # Change indentation slightly
                mut = " " + orig.lstrip()  # reduce indent by 1
                corrupted = prose_replace(corrupted, orig, mut)
                errors.append({"id": f"e_md_sup_{len(errors)}", "category": "markdown_structure", "original_snippet": orig, "mutated_snippet": mut, "rationale": "inconsistent list indent"})
                continue
            # No more safe structural changes available
            break

        # list_structure supplements
        for _ in range(need("list_structure")):
            m = re.search(r"^\-\s+\S", corrupted, flags=re.M)
            if not m:
                break
            orig = m.group(0)
            mut = orig.replace("- ", "-", 1)
            corrupted = prose_replace(corrupted, orig, mut)
            errors.append({"id": f"e_list_sup_{len(errors)}", "category": "list_structure", "original_snippet": orig, "mutated_snippet": mut, "rationale": "bullet missing space"})

        # section_title supplements
        for _ in range(need("section_title")):
            m = re.search(r"^##\s+(What is it\?|What can it do\?|Requirements|Install|Quick example|Learn more|License & Contact)$", corrupted, flags=re.M)
            if not m:
                break
            orig = m.group(0)
            mut = orig.replace("What is it?", "What is It?").replace("Install", "Installation")
            if mut == orig:
                break
            corrupted = prose_replace(corrupted, orig, mut)
            errors.append({"id": f"e_title_sup_{len(errors)}", "category": "section_title", "original_snippet": orig, "mutated_snippet": mut, "rationale": "subtle title change"})

        # image_syntax supplements
        for _ in range(need("image_syntax")):
            m = re.search(r"!\[[^\]]*\]\([^\)]+\)", corrupted)
            if not m:
                break
            orig = m.group(0)
            mut = orig.replace("](", "] (")
            corrupted = prose_replace(corrupted, orig, mut)
            errors.append({"id": f"e_img_sup_{len(errors)}", "category": "image_syntax", "original_snippet": orig, "mutated_snippet": mut, "rationale": "broken image spacing"})

        # inline_code supplements
        # NOTE: Only match single-backtick inline code, NOT code fences or RMarkdown chunks
        _ic_skip = sorted(fence_spans + ([yaml_span] if yaml_span else []))
        for _ in range(need("inline_code")):
            # Find the first backtick-enclosed span that sits outside YAML and fences
            m = next(
                (m for m in re.finditer(r"(?<!`)(?<!^)`([^`\n]{2,30})`(?!`)", corrupted)
                 if not any(s <= m.start() < e for s, e in _ic_skip)),
                None,
            )
            if not m:
                break
            orig = m.group(0)
            inner = m.group(1)
            if inner.startswith("{") or inner.startswith("```"):
                continue
            mut = inner  # Remove surrounding backticks
            new_corrupted = prose_replace(corrupted, orig, mut)
            if new_corrupted == corrupted:
                break  # prose_replace rejected it (e.g. still inside a skip span)
            corrupted = new_corrupted
            errors.append({"id": f"e_code_sup_{len(errors)}", "category": "inline_code", "original_snippet": orig, "mutated_snippet": mut, "rationale": "removed inline code backticks"})

        # ============================================================
        # PROSE-CODE CONSISTENCY supplements (anchor-required)
        # ============================================================

        prose_view = self._prose_region(corrupted)
        code_view = self._extract_code_fragments(baseline)

        # prose_code_param — run BEFORE generic number supplements to claim anchored values
        if need("prose_code_param") > 0:
            param_hit = _PARAM_ANCHOR_RE.search(code_view)
            if param_hit is None:
                self._record_skip(data, "prose_code_param", "no_anchor")
            else:
                param_name = param_hit.group(1)
                code_val = param_hit.group(2)
                mp = re.search(rf"{re.escape(param_name)}\s+(?:of\s+|=\s*)?({re.escape(code_val)})\b", prose_view, re.I)
                if mp is None:
                    self._record_skip(data, "prose_code_param", "no_prose_value_match")
                else:
                    orig = mp.group(0)
                    try:
                        num = float(code_val)
                        new_val = f"{num + 0.1:.1f}" if "." in code_val else str(int(num) + 1)
                    except ValueError:
                        new_val = code_val + "x"
                    mut = orig.replace(code_val, new_val, 1)
                    if orig in corrupted and orig not in corrupted_snippets and mut != orig:
                        corrupted = prose_replace(corrupted, orig, mut)
                        add_error("prose_code_param", orig, mut, f"prose {param_name} drifted from code-used {code_val}")

        # prose_code_pkg_version
        if need("prose_code_pkg_version") > 0:
            pkg_anchor = _PKG_ANCHOR_RE.search(code_view)
            if pkg_anchor is None:
                self._record_skip(data, "prose_code_pkg_version", "no_anchor")
            else:
                pkg_name = pkg_anchor.group(1)
                code_ver = int(pkg_anchor.group(2))
                mp = re.search(rf"\b({re.escape(pkg_name)})\s+v?(\d+)(\.\d+)?\b", self._prose_region(corrupted), re.I)
                if mp is None or int(mp.group(2)) != code_ver:
                    self._record_skip(data, "prose_code_pkg_version", "no_prose_version_match")
                else:
                    orig = mp.group(0)
                    new_v = code_ver - 1 if code_ver > 1 else code_ver + 1
                    mut = f"{mp.group(1)} v{new_v}"
                    if orig in corrupted and orig not in corrupted_snippets:
                        corrupted = prose_replace(corrupted, orig, mut)
                        add_error("prose_code_pkg_version", orig, mut, f"prose {pkg_name} version drifted from code-pinned v{code_ver}")

        # prose_code_stat_test — run early so the prose test name isn't consumed by generic supplements
        if need("prose_code_stat_test") > 0:
            stat_anchor = _STAT_TEST_ANCHOR_RE.search(code_view)
            if stat_anchor is None:
                self._record_skip(data, "prose_code_stat_test", "no_anchor")
            else:
                hit = stat_anchor.group(0).lower()
                # Seurat's FindMarkers / FindAllMarkers default to Wilcoxon.
                code_test = "wilcoxon" if ("wilcox" in hit or "findmarkers" in hit or "findallmarkers" in hit) else "t-test"
                prose_now = self._prose_region(corrupted)
                # Find the prose test NAME THAT MATCHES CODE so we can mutate it to the opposite
                # (this creates the injected disagreement). Skip if prose already disagrees.
                if code_test == "wilcoxon":
                    mp = re.search(r"\b(Wilcoxon(?: rank[- ]sum)?|Mann[- ]Whitney)\b", prose_now, re.I)
                    target = "t-test"
                else:
                    mp = re.search(r"\b(t[- ]test|Student'?s t|two-sample t)\b", prose_now, re.I)
                    target = "Wilcoxon"
                if mp is None:
                    self._record_skip(data, "prose_code_stat_test", "no_prose_test_match")
                else:
                    orig = mp.group(0)
                    if orig in corrupted and orig not in corrupted_snippets:
                        corrupted = prose_replace(corrupted, orig, target)
                        add_error("prose_code_stat_test", orig, target, f"mutated prose '{orig}' so it contradicts code-called {code_test}")

        # prose_code_marker — run early so gene_case / bio_term don't consume the prose marker first
        if need("prose_code_marker") > 0:
            marker_hit = _MARKER_ANCHOR_RE.search(code_view)
            if marker_hit is None:
                self._record_skip(data, "prose_code_marker", "no_anchor")
            else:
                # Group 1: features=c("X"); Group 2: subset(..., X >); Group 3: FeaturePlot/VlnPlot/...
                code_marker = marker_hit.group(1) or marker_hit.group(2) or marker_hit.group(3)
                swap_map = {"CD4": "CD8", "CD8": "CD4", "FOXP3": "RORC", "GATA3": "TBX21"}
                if code_marker not in swap_map:
                    self._record_skip(data, "prose_code_marker", "unmapped_code_marker")
                else:
                    prose_now = self._prose_region(corrupted)
                    mp = re.search(rf"\b{re.escape(code_marker)}\b", prose_now)
                    if mp is None:
                        self._record_skip(data, "prose_code_marker", "no_prose_marker_match")
                    else:
                        orig = mp.group(0)
                        mut = swap_map[code_marker]
                        if orig in corrupted and orig not in corrupted_snippets:
                            corrupted = prose_replace(corrupted, orig, mut)
                            add_error("prose_code_marker", orig, mut, f"mutated prose marker {orig}->{mut} so it contradicts code-referenced {code_marker}")

        # accession_id_prefix — run early so generic number/gene_case don't touch GSE/GSM digits
        if need("accession_id_prefix") > 0:
            prose_now = self._prose_region(corrupted)
            acc_match = None
            for m in re.finditer(r"\b(GSE|GSM)(\d{3,})\b", prose_now):
                ctx_start = max(0, m.start() - 80)
                ctx_end = min(len(prose_now), m.end() + 80)
                if any(re.search(rf"\b{w}\b", prose_now[ctx_start:ctx_end], re.I) for w in _ACCESSION_CONTEXT_WORDS):
                    acc_match = m
                    break
            if acc_match is None:
                self._record_skip(data, "accession_id_prefix", "no_context_anchor")
            else:
                prefix = acc_match.group(1)
                digits = acc_match.group(2)
                orig = prefix + digits
                mut = ("GSM" if prefix == "GSE" else "GSE") + digits
                if orig in corrupted and orig not in corrupted_snippets:
                    corrupted = prose_replace(corrupted, orig, mut)
                    add_error("accession_id_prefix", orig, mut, "swapped accession namespace against surrounding context word")

        # number supplements - change numeric values
        number_attempts = 0
        while need("number") > 0 and number_attempts < min_per_category * 2:
            number_attempts += 1
            found = False
            # Match numbers not in code blocks (simple heuristic)
            for m in re.finditer(r"(?<![`{])\b(\d+\.?\d*)\b(?![`}])", corrupted):
                orig = m.group(0)
                if orig in corrupted_snippets:
                    continue
                # Change the number slightly
                try:
                    num = float(orig)
                    if num > 1:
                        mut = str(int(num) + 1) if "." not in orig else str(num + 0.1)
                    else:
                        mut = str(num * 2) if num != 0 else "1"
                except (ValueError, TypeError):
                    continue
                if mut == orig or mut in corrupted_snippets:
                    continue
                corrupted = prose_replace(corrupted, orig, mut)
                if add_error("number", orig, mut, "changed numeric value"):
                    found = True
                    break
            if not found:
                break

        # boolean supplements - change TRUE/FALSE values
        bool_patterns = [
            (r"\bTRUE\b", "FALSE"),
            (r"\bFALSE\b", "TRUE"),
            (r"\btrue\b", "false"),
            (r"\bfalse\b", "true"),
            (r"\bTrue\b", "False"),
            (r"\bFalse\b", "True"),
        ]
        for pat, replacement in bool_patterns:
            if need("boolean") <= 0:
                break
            m = re.search(pat, corrupted)
            if m:
                orig = m.group(0)
                if orig in corrupted_snippets:
                    continue
                mut = replacement
                corrupted = prose_replace(corrupted, orig, mut)
                add_error("boolean", orig, mut, "flipped boolean value")

        # gene_case supplements - change gene symbol case (important in bioinformatics)
        gene_patterns = [
            (r"\b([A-Z]{2,}[0-9]*)\b", lambda m: m.group(1).lower()),  # BRCA1 -> brca1
            (r"\b([a-z]{2,}[0-9]*)\b", lambda m: m.group(1).upper()),  # brca1 -> BRCA1
        ]
        gene_attempts = 0
        while need("gene_case") > 0 and gene_attempts < min_per_category:
            gene_attempts += 1
            found = False
            # Look for gene-like patterns (2+ letters, possibly followed by numbers)
            for m in re.finditer(r"\b([A-Z]{2,6}[0-9]{0,2})\b", corrupted):
                orig = m.group(0)
                if orig in corrupted_snippets or len(orig) < 3:
                    continue
                # Skip common words that aren't genes
                if orig.lower() in ["the", "and", "for", "not", "are", "was", "rmd", "csv", "pdf"]:
                    continue
                mut = orig.lower()
                if mut == orig or mut in corrupted_snippets:
                    continue
                corrupted = prose_replace(corrupted, orig, mut)
                if add_error("gene_case", orig, mut, "changed gene symbol case"):
                    found = True
                    break
            if not found:
                break

        # param_name supplements - corrupt parameter/argument names (prose only)
        param_attempts = 0
        while need("param_name") > 0 and param_attempts < min_per_category * 2:
            param_attempts += 1
            found = False
            # Search prose region only to avoid matching code-block arg names
            prose_view = self._prose_region(corrupted)
            for m in re.finditer(r"\b([a-z_][a-z0-9_.]*)\s*=\s*", prose_view, flags=re.I):
                param = m.group(1)
                orig = param
                if orig in corrupted_snippets or len(param) < 3:
                    continue
                # Typo the parameter name
                if len(param) > 3:
                    mut = param[:-1]
                else:
                    mut = param + "x"
                if mut == orig or mut in corrupted_snippets:
                    continue
                # Replace in prose only
                full_orig = m.group(0)
                full_mut = full_orig.replace(param, mut, 1)
                new_corrupted = prose_replace(corrupted, full_orig, full_mut)
                if new_corrupted != corrupted:
                    corrupted = new_corrupted
                    if add_error("param_name", orig, mut, "misspelled parameter name"):
                        found = True
                        break
            if not found:
                break

        # comment_typo supplements - typos in R comments (# lines)
        comment_attempts = 0
        while need("comment_typo") > 0 and comment_attempts < min_per_category:
            comment_attempts += 1
            found = False
            # Find comment lines
            for m in re.finditer(r"^#\s*(.+)$", corrupted, flags=re.M):
                comment_text = m.group(1)
                # Find a word in the comment to corrupt
                for word_m in re.finditer(r"\b([A-Za-z]{5,})\b", comment_text):
                    word = word_m.group(1)
                    if word in corrupted_snippets:
                        continue
                    mut = word[:-1]  # Truncate
                    if mut == word or mut in corrupted_snippets:
                        continue
                    corrupted = prose_replace(corrupted, word, mut)
                    if add_error("comment_typo", word, mut, "typo in comment"):
                        found = True
                        break
                if found:
                    break
            if not found:
                break

        # species_name supplements - corrupt species names
        species_swaps = [
            ("human", "humna"),
            ("mouse", "mosue"),
            ("Homo sapiens", "Homo sapien"),
            ("Mus musculus", "Mus musclus"),
        ]
        for orig_sp, mut_sp in species_swaps:
            if need("species_name") <= 0:
                break
            if orig_sp in corrupted and orig_sp not in corrupted_snippets:
                corrupted = prose_replace(corrupted, orig_sp, mut_sp)
                add_error("species_name", orig_sp, mut_sp, "misspelled species name")

        ft = file_type.lower()

        # ── code_func_name supplements (prose .md/.rst/.txt files only) ──────
        if ft in _PROSE_CODE_EXTENSIONS:
            cfn_attempts = 0
            while need("code_func_name") > 0 and cfn_attempts < min_per_category * 2:
                cfn_attempts += 1
                found = False
                code_view = self._extract_code_fragments(corrupted)
                for m in _FUNC_IN_CODE_RE.finditer(code_view):
                    name = m.group(1)
                    if name.lower() in _CODE_SKIP_NAMES or len(name) < 5:
                        continue
                    orig_call = m.group(0)
                    if orig_call in corrupted_snippets:
                        continue
                    mutated = self._transpose_name(name)
                    if mutated == name:
                        continue
                    mut_call = mutated + orig_call[len(name):]
                    if mut_call in corrupted_snippets:
                        continue
                    new_corrupted = fence_replace(corrupted, orig_call, mut_call)
                    if new_corrupted != corrupted:
                        corrupted = new_corrupted
                        if add_error("code_func_name", orig_call.rstrip(), mut_call.rstrip(),
                                     "transposed characters in function name inside code fence"):
                            found = True
                            break
                if not found:
                    break

            # ── code_func_args supplements ────────────────────────────────────
            cfa_attempts = 0
            while need("code_func_args") > 0 and cfa_attempts < min_per_category * 2:
                cfa_attempts += 1
                found = False
                code_view = self._extract_code_fragments(corrupted)
                for m in _NAMED_ARG_IN_CODE_RE.finditer(code_view):
                    name = m.group(1)
                    if name.lower() in _CODE_SKIP_NAMES or len(name) < 3:
                        continue
                    orig_arg = m.group(0)
                    if orig_arg in corrupted_snippets:
                        continue
                    mutated = self._transpose_name(name)
                    if mutated == name:
                        continue
                    mut_arg = mutated + orig_arg[len(name):]
                    if mut_arg in corrupted_snippets:
                        continue
                    new_corrupted = fence_replace(corrupted, orig_arg, mut_arg)
                    if new_corrupted != corrupted:
                        corrupted = new_corrupted
                        if add_error("code_func_args", orig_arg.rstrip(), mut_arg.rstrip(),
                                     "transposed characters in named argument inside code fence"):
                            found = True
                            break
                if not found:
                    break

        # ── code_comment_conflict supplements ─────────────────────────────────
        if ft not in _EXECUTABLE_CODE_EXTENSIONS:
            ccc_attempts = 0
            while need("code_comment_conflict") > 0 and ccc_attempts < min_per_category * 2:
                ccc_attempts += 1
                found = False
                code_view = self._extract_code_fragments(corrupted)
                for m in _COMMENT_IN_CODE_RE.finditer(code_view):
                    prefix = m.group(1)
                    comment_body = m.group(2)
                    for pat, replacement in _COMMENT_CONFLICT_MAP:
                        if not pat.search(comment_body):
                            continue
                        mutated_body = pat.sub(replacement, comment_body, count=1)
                        if mutated_body == comment_body:
                            continue
                        orig_line = prefix + comment_body
                        mut_line = prefix + mutated_body
                        if orig_line in corrupted_snippets or orig_line.rstrip() in corrupted_snippets:
                            continue
                        new_corrupted = fence_replace(corrupted, orig_line, mut_line)
                        if new_corrupted != corrupted:
                            corrupted = new_corrupted
                            if add_error("code_comment_conflict", orig_line.rstrip(), mut_line.rstrip(),
                                         "comment changed to conflict with adjacent code"):
                                found = True
                                break
                    if found:
                        break
                if not found:
                    break

        # ── cli_flag_typo supplements (prose .md/.rst/.txt files only) ────────
        # Pick a long flag from one CLI invocation, permute its stem, then
        # globally rewrite every occurrence (fenced + inline + usage stanzas
        # + prose mentions) so the doc stays internally consistent.
        #
        # Three invariants this loop must hold:
        #   1) Skip flags that are bogus (members of ``_CLI_BOGUS_FLAGS`` — they
        #      were injected by cli_unknown_flag, not real CLI options).
        #   2) Skip flags that any earlier injection step already touched, in
        #      either direction (original or mutated form).  Otherwise the
        #      picker will pick the just-mutated form on the next iteration
        #      and ``_permute_long_flag`` will helpfully transpose it BACK to
        #      the original, silently reversing the prior mutation.
        #   3) Apply ``corrupted = new_corrupted`` only after ``add_error``
        #      succeeds — applying it earlier means an entry that fails the
        #      duplicate-snippet check still mutates the document silently.
        if ft in _PROSE_CODE_EXTENSIONS:
            # Seed the "do-not-touch" set:
            off_limits: Set[str] = {f for f, _ in _CLI_BOGUS_FLAGS}
            for e in errors:
                if e.get("category") in ("cli_flag_typo", "cli_unknown_flag"):
                    for k in ("original_token", "mutated_token"):
                        v = e.get(k)
                        if v:
                            off_limits.add(v)
            cft_attempts = 0
            while need("cli_flag_typo") > 0 and cft_attempts < min_per_category * 2:
                cft_attempts += 1
                found = False
                for line, _source in self._cli_candidates(corrupted):
                    _prefix, tokens = self._split_cli_line(line)
                    picked = self._pick_long_flag(tokens)
                    if not picked:
                        continue
                    _idx, orig_token = picked
                    orig_flag = orig_token.split("=", 1)[0]
                    if orig_flag in off_limits:
                        continue
                    mut_token = self._permute_long_flag(orig_token)
                    if mut_token == orig_token:
                        continue
                    mut_flag = mut_token.split("=", 1)[0]
                    if mut_flag in off_limits:
                        continue
                    new_corrupted, n_replaced = self._replace_flag_globally(
                        corrupted, orig_flag, mut_flag
                    )
                    if n_replaced == 0 or new_corrupted == corrupted:
                        continue
                    mut_line = self._replace_token_in_line(line, orig_token, mut_token) or line
                    if add_error("cli_flag_typo", line, mut_line,
                                 f"permuted long flag {orig_flag} -> {mut_flag} ({n_replaced} occurrences)",
                                 mutated_token=mut_flag):
                        corrupted = new_corrupted
                        fence_spans = self._fence_spans(corrupted)
                        errors[-1]["original_token"] = orig_flag
                        errors[-1]["occurrences_changed"] = n_replaced
                        off_limits.add(orig_flag)
                        off_limits.add(mut_flag)
                        found = True
                        break
                if not found:
                    break

        # ── cli_unknown_flag supplements (prose .md/.rst/.txt files only) ─────
        # As in the primary path: after each shell-line append, also splice
        # the bogus flag into any help block so the corrupted doc remains
        # self-consistent in its wrongness.  Subsequent iterations are
        # idempotent on the help-block surface — the splicer no-ops if the
        # flag is already listed there.
        if ft in _PROSE_CODE_EXTENSIONS:
            cuf_attempts = 0
            while need("cli_unknown_flag") > 0 and cuf_attempts < min_per_category * 2:
                cuf_attempts += 1
                found = False
                bogus_flag, bogus_val = _CLI_BOGUS_FLAGS[
                    cat_counts.get("cli_unknown_flag", 0) % len(_CLI_BOGUS_FLAGS)
                ]
                for line, source in self._cli_candidates(corrupted):
                    if line in corrupted_snippets:
                        continue
                    _prefix, tokens = self._split_cli_line(line)
                    if not tokens or bogus_flag in tokens:
                        continue
                    mut_line = line.rstrip() + f" {bogus_flag} {bogus_val}"
                    new_corrupted = cli_replace(corrupted, line, mut_line, source)
                    if new_corrupted != corrupted:
                        new_corrupted, n_help_edits = self._splice_bogus_flag_into_help_block(
                            new_corrupted, bogus_flag, bogus_val
                        )
                        corrupted = new_corrupted
                        rationale = f"appended bogus flag {bogus_flag}"
                        if n_help_edits:
                            rationale += f" + spliced into help block ({n_help_edits} surface)"
                        if add_error("cli_unknown_flag", line, mut_line, rationale,
                                     mutated_token=bogus_flag):
                            errors[-1]["help_block_edits"] = n_help_edits
                            found = True
                            break
                if not found:
                    break

        # ── cli_program_rename supplements (prose .md/.rst/.txt files only) ───
        if ft in _PROSE_CODE_EXTENSIONS:
            cpr_attempts = 0
            while need("cli_program_rename") > 0 and cpr_attempts < min_per_category * 2:
                cpr_attempts += 1
                found = False
                for line, source in self._cli_candidates(corrupted):
                    if line in corrupted_snippets:
                        continue
                    _prefix, tokens = self._split_cli_line(line)
                    picked = self._pick_program_token(tokens)
                    if not picked:
                        continue
                    _idx, orig_token = picked
                    mut_token = self._mutate_program_token(orig_token)
                    if mut_token == orig_token:
                        continue
                    mut_line = self._replace_token_in_line(line, orig_token, mut_token)
                    if mut_line is None:
                        continue
                    new_corrupted = cli_replace(corrupted, line, mut_line, source)
                    if new_corrupted != corrupted:
                        corrupted = new_corrupted
                        if add_error("cli_program_rename", line, mut_line,
                                     f"mutated program {orig_token} -> {mut_token}",
                                     mutated_token=mut_token):
                            found = True
                            break
                if not found:
                    break

        # ── rd_func_name supplements (only for .Rd files) ─────────────────────
        if self._is_rd_file(corrupted):
            rd_skip = self._rd_skip_spans(corrupted)
            rdfn_attempts = 0
            while need("rd_func_name") > 0 and rdfn_attempts < min_per_category * 2:
                rdfn_attempts += 1
                found = False
                for pattern in (_RD_CODE_FUNC_RE, _RD_LINK_RE):
                    if found:
                        break
                    for m in pattern.finditer(corrupted):
                        if self._in_fence(m.start(), rd_skip):
                            continue
                        name = m.group(1)
                        orig_snippet = m.group(0)
                        if orig_snippet in corrupted_snippets or name in corrupted_snippets:
                            continue
                        if len(name) < 4:
                            continue
                        mutated = self._transpose_name(name)
                        if mutated == name:
                            continue
                        g1s = m.start(1) - m.start()
                        g1e = m.end(1)   - m.start()
                        mut_snippet = orig_snippet[:g1s] + mutated + orig_snippet[g1e:]
                        if mut_snippet in corrupted_snippets:
                            continue
                        new_corrupted = self._replace_in_rd_prose(
                            corrupted, orig_snippet, mut_snippet, rd_skip
                        )
                        if new_corrupted != corrupted:
                            corrupted = new_corrupted
                            rd_skip = self._rd_skip_spans(corrupted)
                            if add_error("rd_func_name", orig_snippet, mut_snippet,
                                         "character swap in function name in .Rd prose"):
                                found = True
                                break
                if not found:
                    break

            # ── rd_arg_name supplements ────────────────────────────────────────
            rdan_attempts = 0
            while need("rd_arg_name") > 0 and rdan_attempts < min_per_category * 2:
                rdan_attempts += 1
                found = False
                # Strategy 1: \code{argname}
                for m in _RD_CODE_ARG_RE.finditer(corrupted):
                    if self._in_fence(m.start(), rd_skip):
                        continue
                    name = m.group(1)
                    orig_snippet = m.group(0)
                    if orig_snippet in corrupted_snippets or name in corrupted_snippets:
                        continue
                    if name.lower() in _CODE_SKIP_NAMES or len(name) < 3:
                        continue
                    mutated = self._transpose_name(name)
                    if mutated == name:
                        continue
                    g1s = m.start(1) - m.start()
                    g1e = m.end(1)   - m.start()
                    mut_snippet = orig_snippet[:g1s] + mutated + orig_snippet[g1e:]
                    if mut_snippet in corrupted_snippets:
                        continue
                    new_corrupted = self._replace_in_rd_prose(
                        corrupted, orig_snippet, mut_snippet, rd_skip
                    )
                    if new_corrupted != corrupted:
                        corrupted = new_corrupted
                        rd_skip = self._rd_skip_spans(corrupted)
                        if add_error("rd_arg_name", orig_snippet, mut_snippet,
                                     "character swap in argument name in .Rd prose"):
                            found = True
                            break
                # Strategy 2: bare argname in \item{arg}{desc}
                if not found:
                    for m in re.finditer(r"\\item\{([a-z][a-z0-9._]{2,})\}\{([^}]+)\}", corrupted):
                        if self._in_fence(m.start(), rd_skip):
                            continue
                        argname = m.group(1)
                        desc = m.group(2)
                        word_m = re.search(rf"\b{re.escape(argname)}\b", desc)
                        if not word_m:
                            continue
                        if argname in corrupted_snippets or argname.lower() in _CODE_SKIP_NAMES or len(argname) < 3:
                            continue
                        mutated = self._transpose_name(argname)
                        if mutated == argname or mutated in corrupted_snippets:
                            continue
                        abs_pos = m.start(2) + word_m.start()
                        new_corrupted = corrupted[:abs_pos] + mutated + corrupted[abs_pos + len(argname):]
                        if new_corrupted != corrupted:
                            corrupted = new_corrupted
                            rd_skip = self._rd_skip_spans(corrupted)
                            if add_error("rd_arg_name", argname, mutated,
                                         f"character swap in argument name '{argname}' in \\item description"):
                                found = True
                                break
                if not found:
                    break

        data["errors"] = errors
        return corrupted, data


