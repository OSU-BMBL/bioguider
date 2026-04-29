from __future__ import annotations

import json
from typing import Tuple, Dict, Any, List, Set
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

EXTERNAL-AUTHORITY ERROR CATEGORIES (inject only when a context anchor exists)
- accession_id_prefix: swap an accession ID prefix so the namespace contradicts a nearby context word (e.g., prose says "series" but mutate `GSE123456` → `GSM123456`, or prose says "samples" but mutate `GSM123456` → `GSE123456`). Only inject when a context word (series/samples/experiment/run/study) appears within the same sentence as the accession ID. Otherwise skip and record in "skipped".

PROSE-CODE CONSISTENCY ERROR CATEGORIES (inject ONLY when a matching code-block anchor exists; otherwise skip and record in "skipped" with reason "no_anchor")
- prose_code_pkg_version: prose narrates a package major version that disagrees with the version pinned/loaded in a fenced code block (e.g., `library(Seurat)` with `sessionInfo()` showing `Seurat_5.0.1`, but prose says "Seurat v4"). Mutate the prose version only; leave code untouched.
- prose_code_stat_test: prose narrates a statistical test name that disagrees with the function actually called in a fenced code block (e.g., code runs `wilcox.test(...)` or `FindMarkers(..., test.use="wilcox")`, but prose says "t-test"). Mutate the prose test name only.
- prose_code_marker: prose names a cell-type marker gene that disagrees with the marker used in a fenced code block (e.g., code subsets on `CD8` via `subset(..., CD8 > 0)` or `features = c("CD8")`, but prose describes "CD4+ T cells"). Mutate the prose marker only.
- prose_code_param: prose states an analysis hyperparameter value that disagrees with the value passed to the corresponding function call in a fenced code block (e.g., code runs `FindClusters(..., resolution = 0.5)`, but prose says "resolution of 0.6"). Mutate the prose value only.


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
    def _in_fence(pos: int, spans: List[Tuple[int, int]]) -> bool:
        """Return True if character position ``pos`` falls inside any fence span."""
        for start, end in spans:
            if start <= pos < end:
                return True
            if start > pos:
                break
        return False

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

    @staticmethod
    def _record_skip(data: Dict[str, Any], category: str, reason: str) -> None:
        data.setdefault("skipped", []).append({"category": category, "reason": reason})

    def inject(self, readme_text: str, min_per_category: int = 3, preserve_keywords: list[str] | None = None, max_words: int = 450, project_terms: list[str] | None = None, force_deterministic: bool | None = None) -> Tuple[str, Dict[str, Any]]:
        # Resolve the per-call override; fall back to instance default.
        use_det = self.force_deterministic if force_deterministic is None else force_deterministic

        if use_det:
            # Skip the LLM entirely — deterministic inject + supplements only.
            corrupted, data = self._deterministic_inject(readme_text)
            corrupted, data = self._supplement_errors(
                readme_text, corrupted, data, min_per_category, project_terms
            )
            if not self._check_code_blocks_preserved(readme_text, corrupted):
                # Supplements broke fences — fall back to bare deterministic output.
                corrupted, data = self._deterministic_inject(readme_text)
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
            corrupted, data = self._deterministic_inject(readme_text)
        # Validate output stays within original context; fallback to deterministic if invalid
        elif not self._validate_corrupted(readme_text, corrupted, preserve_keywords):
            corrupted, data = self._deterministic_inject(readme_text)
            
        # Supplement to satisfy minimum per-category counts using deterministic local edits
        corrupted, data = self._supplement_errors(readme_text, corrupted, data, min_per_category, project_terms)
        
        # Final safety check: ensure code blocks are still intact after supplements
        if not self._check_code_blocks_preserved(readme_text, corrupted):
            print("Warning: Supplements broke code blocks, reverting to baseline with minimal errors")
            corrupted, data = self._deterministic_inject(readme_text)
        
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

    def _deterministic_inject(self, baseline: str) -> Tuple[str, Dict[str, Any]]:
        errors: List[Dict[str, Any]] = []
        text = baseline
        # typo
        if "successfully" in text:
            text = text.replace("successfully", "succesfully", 1)
            errors.append({"id": "e_typo_1", "category": "typo", "original_snippet": "successfully", "mutated_snippet": "succesfully", "rationale": "common misspelling"})
        elif "installation" in text:
            text = text.replace("installation", "instalation", 1)
            errors.append({"id": "e_typo_1", "category": "typo", "original_snippet": "installation", "mutated_snippet": "instalation", "rationale": "common misspelling"})
        # link
        m = re.search(r"\]\(https?://[^)]+\)", text)
        if m:
            broken = m.group(0).replace("https://", "https//")
            text = text.replace(m.group(0), broken, 1)
            errors.append({"id": "e_link_1", "category": "link", "original_snippet": m.group(0), "mutated_snippet": broken, "rationale": "missing colon in scheme"})
        # duplicate a small section (next header and paragraph)
        lines = text.splitlines()
        dup_idx = next((i for i, ln in enumerate(lines) if ln.strip().startswith("## ")), None)
        if dup_idx is not None:
            block = lines[dup_idx: min(len(lines), dup_idx+5)]
            text = "\n".join(lines + ["", *block])
            errors.append({"id": "e_dup_1", "category": "duplicate", "original_snippet": "\n".join(block), "mutated_snippet": "\n".join(block), "rationale": "duplicated section"})
        # markdown structure: break a header
        if "\n# " in text:
            text = text.replace("\n# ", "\n#", 1)
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

        return text, data

    def _supplement_errors(self, baseline: str, corrupted: str, data: Dict[str, Any], min_per_category: int, project_terms: list[str] | None = None) -> Tuple[str, Dict[str, Any]]:
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

        # Pre-compute fence spans for prose-only replacement.
        # Recomputed after each replacement to stay in sync with offsets.
        fence_spans = self._fence_spans(corrupted)

        def need(cat: str) -> int:
            return max(0, min_per_category - cat_counts.get(cat, 0))

        def prose_replace(text: str, old: str, new: str) -> str:
            """Replace first occurrence of ``old`` outside fenced code blocks."""
            nonlocal fence_spans
            result = self._replace_prose_only(text, old, new, fence_spans)
            if result != text:
                fence_spans = self._fence_spans(result)
            return result

        def add_error(cat: str, orig: str, mut: str, rationale: str) -> bool:
            """Add error and update tracking. Returns True if added."""
            if orig in corrupted_snippets or mut in corrupted_snippets:
                return False  # Already corrupted
            errors.append({
                "id": f"e_{cat}_sup_{len(errors)}",
                "category": cat,
                "original_snippet": orig,
                "mutated_snippet": mut,
                "rationale": rationale
            })
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
        for _ in range(need("inline_code")):
            # Match inline code that:
            # - Is NOT at the start of a line (to avoid code fences)
            # - Contains word characters (actual code, not just punctuation)
            # - Is surrounded by single backticks only
            m = re.search(r"(?<!`)(?<!^)`([^`\n]{2,30})`(?!`)", corrupted)
            if not m:
                break
            orig = m.group(0)
            inner = m.group(1)
            # Skip if it looks like a code fence or RMarkdown chunk marker
            if inner.startswith("{") or inner.startswith("```"):
                continue
            mut = inner  # Remove surrounding backticks
            corrupted = prose_replace(corrupted, orig, mut)
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

        # param_name supplements - corrupt parameter/argument names
        param_attempts = 0
        while need("param_name") > 0 and param_attempts < min_per_category * 2:
            param_attempts += 1
            found = False
            # Match parameter assignments like "param = value" or "param=value"
            for m in re.finditer(r"\b([a-z_][a-z0-9_.]*)\s*=\s*", corrupted, flags=re.I):
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
                # Replace in context
                full_orig = m.group(0)
                full_mut = full_orig.replace(param, mut, 1)
                corrupted = prose_replace(corrupted, full_orig, full_mut)
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

        data["errors"] = errors
        return corrupted, data


