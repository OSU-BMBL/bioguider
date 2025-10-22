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


CONSTRAINTS
- Keep edits minimal and local; **≥85% token overlap** with input.
- **Preserve section ORDER and TITLES** from the Intro spec:
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


class LLMErrorInjector:
    def __init__(self, llm: BaseChatOpenAI):
        self.llm = llm

    def inject(self, readme_text: str, min_per_category: int = 3, preserve_keywords: list[str] | None = None, max_words: int = 450) -> Tuple[str, Dict[str, Any]]:
        conv = CommonConversation(self.llm)
        preserve_keywords = preserve_keywords or self._extract_preserve_keywords(readme_text)
        system_prompt = escape_braces(INJECTION_PROMPT).format(
            readme=readme_text[:30000],
            min_per_category=min_per_category,
            keywords=", ".join(preserve_keywords) if preserve_keywords else "",
            max_words=max_words,
        )
        output, _ = conv.generate(system_prompt=system_prompt, instruction_prompt="Return the JSON now.")
        
        # Enhanced JSON parsing with better error handling
        data = self._parse_json_output(output, readme_text)
        corrupted = data.get("corrupted_markdown", readme_text)
        # Validate output stays within original context; fallback to deterministic if invalid
        if not self._validate_corrupted(readme_text, corrupted, preserve_keywords):
            corrupted, data = self._deterministic_inject(readme_text)
        # Supplement to satisfy minimum per-category counts using deterministic local edits
        corrupted, data = self._supplement_errors(readme_text, corrupted, data, min_per_category)
        manifest = {
            "errors": data.get("errors", []),
        }
        return corrupted, manifest

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
        # Similarity threshold
        ratio = SequenceMatcher(None, baseline, corrupted).ratio()
        if ratio < 0.7:
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
        return text, {"errors": errors}

    def _supplement_errors(self, baseline: str, corrupted: str, data: Dict[str, Any], min_per_category: int) -> Tuple[str, Dict[str, Any]]:
        errors: List[Dict[str, Any]] = data.get("errors", []) or []
        cat_counts: Dict[str, int] = {}
        for e in errors:
            cat = e.get("category", "")
            cat_counts[cat] = cat_counts.get(cat, 0) + 1

        def need(cat: str) -> int:
            return max(0, min_per_category - cat_counts.get(cat, 0))

        # typo supplements
        for _ in range(need("typo")):
            m = re.search(r"\b(installation|successfully|analysis|documentation|maintained|example|requirements|license|tutorials)\b", corrupted, flags=re.I)
            if not m:
                m = re.search(r"\b[A-Za-z]{6,}\b", corrupted)
            if not m:
                break
            orig = m.group(0)
            mut = orig[:-1] if len(orig) > 3 else orig + "e"
            corrupted = corrupted.replace(orig, mut, 1)
            errors.append({"id": f"e_typo_sup_{len(errors)}", "category": "typo", "original_snippet": orig, "mutated_snippet": mut, "rationale": "minor misspelling"})

        # link supplements
        for _ in range(need("link")):
            m = re.search(r"\[[^\]]+\]\(https?://[^)]+\)", corrupted)
            if not m:
                break
            orig = m.group(0)
            mut = orig.replace("https://", "https//", 1)
            if mut == orig:
                mut = orig.replace("http://", "http//", 1)
            corrupted = corrupted.replace(orig, mut, 1)
            errors.append({"id": f"e_link_sup_{len(errors)}", "category": "link", "original_snippet": orig, "mutated_snippet": mut, "rationale": "scheme colon removed"})

        # duplicate supplements (cap to min_per_category)
        for _ in range(min(need("duplicate"), min_per_category)):
            lines = corrupted.splitlines()
            idx = next((i for i, ln in enumerate(lines) if ln.strip().startswith("- ") or ln.strip().startswith("## ")), None)
            if idx is None:
                break
            frag = lines[idx]
            lines = lines[:idx+1] + [frag] + lines[idx+1:]
            corrupted = "\n".join(lines)
            errors.append({"id": f"e_dup_sup_{len(errors)}", "category": "duplicate", "original_snippet": frag, "mutated_snippet": frag, "rationale": "line duplicated"})

        # bio_term supplements
        bio_swaps = [(r"single cell", "single sell"), (r"genomics", "genomis"), (r"spatial", "spacial")]
        for _ in range(need("bio_term")):
            made = False
            for pat, rep in bio_swaps:
                m = re.search(pat, corrupted, flags=re.I)
                if m:
                    orig = m.group(0)
                    mut = rep if orig.islower() else rep.title()
                    corrupted = corrupted.replace(orig, mut, 1)
                    errors.append({"id": f"e_bio_sup_{len(errors)}", "category": "bio_term", "original_snippet": orig, "mutated_snippet": mut, "rationale": "common domain typo"})
                    made = True
                    break
            if not made:
                break

        # function supplements
        for _ in range(need("function")):
            m = re.search(r"\b([A-Za-z_][A-Za-z0-9_]*)\(", corrupted)
            if not m:
                break
            fname = m.group(1)
            if len(fname) > 3:
                mut = fname[:-1]
            else:
                mut = fname + "x"
            orig = fname + "("
            mutated = mut + "("
            corrupted = corrupted.replace(orig, mutated, 1)
            errors.append({"id": f"e_func_sup_{len(errors)}", "category": "function", "original_snippet": orig, "mutated_snippet": mutated, "rationale": "misspelled API name"})

        # markdown_structure supplements
        for _ in range(need("markdown_structure")):
            m = re.search(r"^## \s*", corrupted, flags=re.M)
            if m:
                orig = m.group(0)
                mut = orig.replace("## ", "##", 1)
                corrupted = corrupted.replace(orig, mut, 1)
                errors.append({"id": f"e_md_sup_{len(errors)}", "category": "markdown_structure", "original_snippet": orig, "mutated_snippet": mut, "rationale": "removed header space"})
            else:
                fence = re.search(r"```[A-Za-z]*\n[\s\S]*?```", corrupted)
                if fence:
                    block = fence.group(0)
                    mut = block.rstrip("`")  # drop a backtick
                    corrupted = corrupted.replace(block, mut, 1)
                    errors.append({"id": f"e_md_sup_{len(errors)}", "category": "markdown_structure", "original_snippet": block[:10], "mutated_snippet": mut[:10], "rationale": "broken code fence"})
                else:
                    break

        # list_structure supplements
        for _ in range(need("list_structure")):
            m = re.search(r"^\-\s+\S", corrupted, flags=re.M)
            if not m:
                break
            orig = m.group(0)
            mut = orig.replace("- ", "-", 1)
            corrupted = corrupted.replace(orig, mut, 1)
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
            corrupted = corrupted.replace(orig, mut, 1)
            errors.append({"id": f"e_title_sup_{len(errors)}", "category": "section_title", "original_snippet": orig, "mutated_snippet": mut, "rationale": "subtle title change"})

        # image_syntax supplements
        for _ in range(need("image_syntax")):
            m = re.search(r"!\[[^\]]*\]\([^\)]+\)", corrupted)
            if not m:
                break
            orig = m.group(0)
            mut = orig.replace("](", "] (")
            corrupted = corrupted.replace(orig, mut, 1)
            errors.append({"id": f"e_img_sup_{len(errors)}", "category": "image_syntax", "original_snippet": orig, "mutated_snippet": mut, "rationale": "broken image spacing"})

        # inline_code supplements
        for _ in range(need("inline_code")):
            m = re.search(r"`[^`\n]+`", corrupted)
            if not m:
                break
            orig = m.group(0)
            mut = orig.strip("`")
            corrupted = corrupted.replace(orig, mut, 1)
            errors.append({"id": f"e_code_sup_{len(errors)}", "category": "inline_code", "original_snippet": orig, "mutated_snippet": mut, "rationale": "removed inline code backticks"})

        data["errors"] = errors
        return corrupted, data


