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


class LLMErrorInjector:
    def __init__(self, llm: BaseChatOpenAI):
        self.llm = llm

    def inject(self, readme_text: str, min_per_category: int = 3, preserve_keywords: list[str] | None = None, max_words: int = 450, project_terms: list[str] | None = None) -> Tuple[str, Dict[str, Any]]:
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
        return text, {"errors": errors}

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

        def need(cat: str) -> int:
            return max(0, min_per_category - cat_counts.get(cat, 0))
        
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
                    
                    corrupted = corrupted.replace(orig, mut, 1)
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
                    
                    corrupted = corrupted.replace(orig, mut, 1)
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
                corrupted = corrupted.replace(orig, mut, 1)
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
                corrupted = corrupted.replace(orig, mut, 1)
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
                    
                    corrupted = corrupted.replace(orig, mut, 1)
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
                
                corrupted = corrupted.replace(orig, mutated, 1)
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
                    corrupted = corrupted.replace(orig, mut, 1)
                    errors.append({"id": f"e_md_sup_{len(errors)}", "category": "markdown_structure", "original_snippet": orig.strip(), "mutated_snippet": mut.strip(), "rationale": "removed header space"})
                    continue
            # Try list indentation issues (safe)
            m = re.search(r"^( {2,4})[-*]", corrupted, flags=re.M)
            if m:
                orig = m.group(0)
                # Change indentation slightly
                mut = " " + orig.lstrip()  # reduce indent by 1
                corrupted = corrupted.replace(orig, mut, 1)
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
            corrupted = corrupted.replace(orig, mut, 1)
            errors.append({"id": f"e_code_sup_{len(errors)}", "category": "inline_code", "original_snippet": orig, "mutated_snippet": mut, "rationale": "removed inline code backticks"})

        # ============================================================
        # NEW ERROR CATEGORIES for more diverse injection
        # ============================================================
        
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
                except:
                    continue
                if mut == orig or mut in corrupted_snippets:
                    continue
                corrupted = corrupted.replace(orig, mut, 1)
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
                corrupted = corrupted.replace(orig, mut, 1)
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
                corrupted = corrupted.replace(orig, mut, 1)
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
                corrupted = corrupted.replace(full_orig, full_mut, 1)
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
                    corrupted = corrupted.replace(word, mut, 1)
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
                corrupted = corrupted.replace(orig_sp, mut_sp, 1)
                add_error("species_name", orig_sp, mut_sp, "misspelled species name")

        data["errors"] = errors
        return corrupted, data


