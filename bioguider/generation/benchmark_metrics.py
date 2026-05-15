"""
Benchmark metrics for comprehensive error injection evaluation.

Provides F-score calculation with semantic False Positive detection via LLM.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher, unified_diff
from typing import Dict, Any, List, Tuple, Optional

from langchain_openai.chat_models.base import BaseChatOpenAI
from bioguider.agents.common_conversation import CommonConversation
from bioguider.managers.config import (
    UNSCORABLE_CATEGORIES,
    CONTENT_CATEGORIES,
    HYGIENE_CATEGORIES,
    compute_scorable_breakdown,
)
from bioguider.generation.unified_metrics import _naked_count


def check_protected_regions(baseline: str, revised: str) -> dict:
    """Return violation counts for code fences, YAML frontmatter, and section headers."""
    # Code fence comparison
    fence_pattern = r'(```[^\n]*\n.*?```)'
    baseline_fences = re.findall(fence_pattern, baseline, re.DOTALL)
    revised_fences = re.findall(fence_pattern, revised, re.DOTALL)
    if len(baseline_fences) != len(revised_fences):
        code_fence_violations = abs(len(baseline_fences) - len(revised_fences))
    else:
        code_fence_violations = sum(
            1 for b, r in zip(baseline_fences, revised_fences) if b != r
        )

    # YAML frontmatter comparison
    yaml_pattern = r'\A---\n(.*?)\n---'
    baseline_yaml = re.search(yaml_pattern, baseline, re.DOTALL)
    revised_yaml = re.search(yaml_pattern, revised, re.DOTALL)
    if baseline_yaml is None and revised_yaml is None:
        yaml_violations = 0
    elif baseline_yaml is None or revised_yaml is None:
        yaml_violations = 1
    else:
        yaml_violations = 0 if baseline_yaml.group(1) == revised_yaml.group(1) else 1

    # Section header comparison
    header_pattern = r'^#{1,6}\s+.+$'
    baseline_headers = re.findall(header_pattern, baseline, re.MULTILINE)
    revised_headers = re.findall(header_pattern, revised, re.MULTILINE)
    if baseline_headers == revised_headers:
        section_violations = 0
    else:
        # Count differing positions plus any count difference
        common_len = min(len(baseline_headers), len(revised_headers))
        section_violations = abs(len(baseline_headers) - len(revised_headers)) + sum(
            1 for b, r in zip(baseline_headers[:common_len], revised_headers[:common_len]) if b != r
        )

    return {
        "code_fence_violations": code_fence_violations,
        "yaml_violations": yaml_violations,
        "section_violations": section_violations,
    }


def count_collateral_damage(baseline: str, revised: str, errors: list) -> list:
    """Find prose changes outside injected error regions and protected areas."""
    baseline_lines = baseline.splitlines()
    revised_lines = revised.splitlines()

    # Pre-build set of line indices inside code fences or YAML frontmatter
    def _protected_indices(lines: list) -> set:
        protected = set()
        in_fence = False
        in_yaml = False
        for i, line in enumerate(lines):
            stripped = line.strip()
            if i == 0 and stripped == '---':
                in_yaml = True
                protected.add(i)
                continue
            if in_yaml:
                protected.add(i)
                if stripped == '---':
                    in_yaml = False
                continue
            if stripped.startswith('```'):
                protected.add(i)
                in_fence = not in_fence
                continue
            if in_fence:
                protected.add(i)
        return protected

    baseline_protected = _protected_indices(baseline_lines)

    matcher = SequenceMatcher(None, baseline_lines, revised_lines)
    collateral = []

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'equal':
            continue

        changed_baseline = "\n".join(baseline_lines[i1:i2])
        changed_revised = "\n".join(revised_lines[j1:j2])

        # Exclusion 1: overlap with an injected error snippet
        is_error_related = False
        for err in errors:
            orig = err.get("original_snippet", "")
            mut = err.get("mutated_snippet", "")
            if orig and orig in changed_baseline:
                is_error_related = True
                break
            if mut and mut in changed_baseline:
                is_error_related = True
                break
        if is_error_related:
            continue

        # Exclusion 2: all changed baseline lines are inside protected regions
        if i1 < i2 and all(idx in baseline_protected for idx in range(i1, i2)):
            continue

        collateral.append({
            "original": changed_baseline,
            "changed": changed_revised,
        })

    return collateral


@dataclass
class ErrorMetrics:
    """Metrics for a single error evaluation."""
    error_id: str
    category: str
    file_path: str
    is_fixed: bool  # TP if True, FN if False
    original_snippet: str
    mutated_snippet: str
    status: str  # "fixed_to_baseline", "fixed_to_valid", "unchanged"


@dataclass
class FalsePositive:
    """Represents a detected false positive (harmful unintended change)."""
    file_path: str
    change_description: str
    severity: str  # "harmful", "neutral", "beneficial"
    original_text: str
    changed_text: str


@dataclass
class BenchmarkResult:
    """Complete benchmark result for a single run."""
    error_count: int
    file_count: int
    
    # Core metrics
    true_positives: int = 0  # Errors correctly fixed
    false_negatives: int = 0  # Errors NOT fixed
    false_positives: int = 0  # Harmful unintended changes
    true_negatives: int = 0  # Non-errors correctly unchanged
    
    # Derived metrics (computed)
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    fix_rate: float = 0.0

    # Scorable-only metrics — UNSCORABLE_CATEGORIES excluded. Headline numbers
    # for the paper figure; the `function` category is injected but not scored
    # because BioGuider's locator uses function names as anchors (structurally
    # unfixable by design).
    true_positives_scorable: int = 0
    false_negatives_scorable: int = 0
    false_positives_scorable: int = 0
    total_errors_scorable: int = 0
    precision_scorable: float = 0.0
    recall_scorable: float = 0.0
    f1_score_scorable: float = 0.0
    fix_rate_scorable: float = 0.0

    # Paper-table CONTENT vs HYGIENE split.
    total_injected_content: int = 0
    fixed_content: int = 0
    f1_score_content: float = 0.0
    total_injected_hygiene: int = 0
    fixed_hygiene: int = 0
    f1_score_hygiene: float = 0.0

    # Protected region violations (Hard FP) from check_protected_regions()
    code_fence_violations: int = 0
    yaml_violations: int = 0
    section_violations: int = 0

    # Detailed breakdowns
    per_category: Dict[str, Dict[str, int]] = field(default_factory=dict)
    per_file: Dict[str, Dict[str, int]] = field(default_factory=dict)
    error_details: List[ErrorMetrics] = field(default_factory=list)
    fp_details: List[FalsePositive] = field(default_factory=list)

    def compute_derived_metrics(self):
        """Compute precision, recall, F1 from TP/FP/FN."""
        # Precision = TP / (TP + FP)
        if self.true_positives + self.false_positives > 0:
            self.precision = self.true_positives / (self.true_positives + self.false_positives)
        else:
            self.precision = 0.0

        # Recall = TP / (TP + FN)
        if self.true_positives + self.false_negatives > 0:
            self.recall = self.true_positives / (self.true_positives + self.false_negatives)
        else:
            self.recall = 0.0

        # F1 = 2 * (precision * recall) / (precision + recall)
        if self.precision + self.recall > 0:
            self.f1_score = 2 * (self.precision * self.recall) / (self.precision + self.recall)
        else:
            self.f1_score = 0.0

        # Fix rate = TP / (TP + FN)
        total_errors = self.true_positives + self.false_negatives
        if total_errors > 0:
            self.fix_rate = self.true_positives / total_errors
        else:
            self.fix_rate = 0.0

        # Scorable-only metrics — shared helper so stress-test figures and
        # unified_metrics stay in sync on the UNSCORABLE_CATEGORIES carve-out.
        b = compute_scorable_breakdown(
            self.error_details,
            false_positives_total=self.false_positives,
        )
        self.true_positives_scorable = b["tp_scorable"]
        self.false_negatives_scorable = b["fn_scorable"]
        self.false_positives_scorable = b["fp_scorable"]
        self.total_errors_scorable = b["total_scorable"]
        self.precision_scorable = b["precision_scorable"]
        self.recall_scorable = b["recall_scorable"]
        self.f1_score_scorable = b["f1_score_scorable"]
        self.fix_rate_scorable = b["fix_rate_scorable"]

        # CONTENT vs HYGIENE split. Precision mirrors the scorable precision
        # (FPs are not attributed to a group); recall is group-local.
        fixed_c = sum(
            1 for e in self.error_details
            if e.category in CONTENT_CATEGORIES and e.is_fixed
        )
        total_c = sum(
            1 for e in self.error_details
            if e.category in CONTENT_CATEGORIES
        )
        fixed_h = sum(
            1 for e in self.error_details
            if e.category in HYGIENE_CATEGORIES and e.is_fixed
        )
        total_h = sum(
            1 for e in self.error_details
            if e.category in HYGIENE_CATEGORIES
        )
        recall_c = fixed_c / total_c if total_c > 0 else 0.0
        recall_h = fixed_h / total_h if total_h > 0 else 0.0
        prec_s = self.precision_scorable
        self.total_injected_content = total_c
        self.fixed_content = fixed_c
        self.f1_score_content = (
            2 * prec_s * recall_c / (prec_s + recall_c)
            if (prec_s + recall_c) > 0 else 0.0
        )
        self.total_injected_hygiene = total_h
        self.fixed_hygiene = fixed_h
        self.f1_score_hygiene = (
            2 * prec_s * recall_h / (prec_s + recall_h)
            if (prec_s + recall_h) > 0 else 0.0
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "error_count": self.error_count,
            "file_count": self.file_count,
            "true_positives": self.true_positives,
            "false_negatives": self.false_negatives,
            "false_positives": self.false_positives,
            "true_negatives": self.true_negatives,
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1_score": round(self.f1_score, 4),
            "fix_rate": round(self.fix_rate, 4),
            "unscorable_categories": sorted(UNSCORABLE_CATEGORIES),
            "total_errors_scorable": self.total_errors_scorable,
            "true_positives_scorable": self.true_positives_scorable,
            "false_negatives_scorable": self.false_negatives_scorable,
            "false_positives_scorable": self.false_positives_scorable,
            "precision_scorable": round(self.precision_scorable, 4),
            "recall_scorable": round(self.recall_scorable, 4),
            "f1_score_scorable": round(self.f1_score_scorable, 4),
            "fix_rate_scorable": round(self.fix_rate_scorable, 4),
            "total_injected_content": self.total_injected_content,
            "fixed_content": self.fixed_content,
            "f1_score_content": round(self.f1_score_content, 4),
            "total_injected_hygiene": self.total_injected_hygiene,
            "fixed_hygiene": self.fixed_hygiene,
            "f1_score_hygiene": round(self.f1_score_hygiene, 4),
            "per_category": self.per_category,
            "per_file": self.per_file,
            "error_details": [
                {
                    "error_id": e.error_id,
                    "category": e.category,
                    "file_path": e.file_path,
                    "is_fixed": e.is_fixed,
                    "status": e.status,
                }
                for e in self.error_details
            ],
            "fp_details": [
                {
                    "file_path": fp.file_path,
                    "change_description": fp.change_description,
                    "severity": fp.severity,
                }
                for fp in self.fp_details
            ],
        }


SEMANTIC_FP_PROMPT = """
You are analyzing changes made to a documentation file to detect potentially harmful modifications.

CONTEXT:
- A document was intentionally corrupted with specific errors (listed below)
- An AI system attempted to fix these errors
- We need to check if the AI made any UNINTENDED harmful changes beyond fixing the known errors

INJECTED ERRORS (these changes ARE expected and should be fixed):
{injected_errors}

DIFF OF CHANGES (unified diff format):
```
{diff}
```

TASK:
Analyze the diff and identify any changes that are NOT related to fixing the injected errors.
For each unrelated change, classify it as:
1. "harmful" - Incorrect changes that introduce new errors or break functionality
2. "neutral" - Style/formatting changes that don't affect correctness
3. "beneficial" - Improvements beyond the required fixes (still acceptable)

OUTPUT (JSON only):
{{
  "unintended_changes": [
    {{
      "description": "brief description of the change",
      "severity": "harmful|neutral|beneficial",
      "original_text": "what was there before",
      "changed_text": "what it was changed to",
      "reasoning": "why this classification"
    }}
  ],
  "summary": {{
    "harmful_count": <int>,
    "neutral_count": <int>,
    "beneficial_count": <int>
  }}
}}

If no unintended changes found, return:
{{
  "unintended_changes": [],
  "summary": {{"harmful_count": 0, "neutral_count": 0, "beneficial_count": 0}}
}}
"""


class SemanticFPDetector:
    """Detects false positives using LLM semantic analysis."""
    
    def __init__(self, llm: BaseChatOpenAI):
        self.llm = llm
    
    def detect_false_positives(
        self,
        baseline: str,
        revised: str,
        injected_errors: List[Dict[str, Any]],
        file_path: str
    ) -> List[FalsePositive]:
        """
        Detect harmful unintended changes (false positives) in the revised content.
        
        Args:
            baseline: Original correct content
            revised: Content after AI fixes
            injected_errors: List of errors that were intentionally injected
            file_path: Path to the file being analyzed
            
        Returns:
            List of detected false positives (harmful changes)
        """
        # Generate unified diff
        baseline_lines = baseline.splitlines(keepends=True)
        revised_lines = revised.splitlines(keepends=True)
        diff_lines = list(unified_diff(
            baseline_lines, 
            revised_lines, 
            fromfile="baseline", 
            tofile="revised",
            lineterm=""
        ))
        diff_text = "".join(diff_lines)
        
        if not diff_text.strip():
            # No changes at all
            return []
        
        # Format injected errors for the prompt
        error_descriptions = []
        for err in injected_errors:
            error_descriptions.append(
                f"- Category: {err.get('category', 'unknown')}\n"
                f"  Original: {err.get('original_snippet', 'N/A')[:100]}\n"
                f"  Mutated: {err.get('mutated_snippet', 'N/A')[:100]}"
            )
        errors_text = "\n".join(error_descriptions) if error_descriptions else "None"
        
        # Build prompt
        prompt = SEMANTIC_FP_PROMPT.format(
            injected_errors=errors_text,
            diff=diff_text[:8000]  # Limit diff size
        )
        
        try:
            conv = CommonConversation(self.llm)
            output, _ = conv.generate(
                system_prompt=prompt,
                instruction_prompt="Analyze the changes and return the JSON."
            )
            
            # Parse response
            result = self._parse_json_output(output)
            
            # Extract harmful changes as false positives
            false_positives = []
            for change in result.get("unintended_changes", []):
                if change.get("severity") == "harmful":
                    false_positives.append(FalsePositive(
                        file_path=file_path,
                        change_description=change.get("description", "Unknown change"),
                        severity="harmful",
                        original_text=change.get("original_text", ""),
                        changed_text=change.get("changed_text", ""),
                    ))
            
            return false_positives
            
        except Exception as e:
            print(f"Warning: Semantic FP detection failed for {file_path}: {e}")
            return []
    
    def _parse_json_output(self, output: str) -> Dict[str, Any]:
        """Parse JSON from LLM output with fallback strategies."""
        # Strategy 1: Direct parse
        try:
            return json.loads(output)
        except json.JSONDecodeError:
            pass
        
        # Strategy 2: Extract JSON block
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
            
            if brace_count == 0:
                try:
                    return json.loads(output[start:end+1])
                except json.JSONDecodeError:
                    pass
        
        # Fallback
        return {"unintended_changes": [], "summary": {"harmful_count": 0, "neutral_count": 0, "beneficial_count": 0}}


class BenchmarkEvaluator:
    """Evaluates benchmark results with F-score metrics."""
    
    def __init__(self, llm: Optional[BaseChatOpenAI] = None):
        self.llm = llm
        self.fp_detector = SemanticFPDetector(llm) if llm else None
    
    def evaluate_single_file(
        self,
        baseline: str,
        corrupted: str,
        revised: str,
        injection_manifest: Dict[str, Any],
        file_path: str,
        file_category: str,
        detect_semantic_fp: bool = True
    ) -> Tuple[List[ErrorMetrics], List[FalsePositive]]:
        """
        Evaluate fixes for a single file.
        
        Returns:
            Tuple of (error_metrics, false_positives)
        """
        error_metrics = []
        
        for err in injection_manifest.get("errors", []):
            error_id = err.get("id", "unknown")
            category = err.get("category", "unknown")
            orig = err.get("original_snippet", "")
            mut = err.get("mutated_snippet", "")
            mutated_token = err.get("mutated_token", "")

            # Determine if error was fixed
            is_fixed, status = self._check_error_fixed(
                category, orig, mut, baseline, corrupted, revised,
                mutated_token=mutated_token,
            )
            
            error_metrics.append(ErrorMetrics(
                error_id=error_id,
                category=category,
                file_path=file_path,
                is_fixed=is_fixed,
                original_snippet=orig,
                mutated_snippet=mut,
                status=status,
            ))
        
        # Deterministic FP: collateral damage (always computed, no LLM needed)
        false_positives = []
        collateral_changes = count_collateral_damage(
            baseline, revised, injection_manifest.get("errors", [])
        )
        for change in collateral_changes:
            false_positives.append(FalsePositive(
                file_path=file_path,
                change_description="Collateral prose change",
                severity="harmful",
                original_text=change["original"][:200],
                changed_text=change["changed"][:200],
            ))

        # Semantic FP: optional, needs LLM (additive)
        if detect_semantic_fp and self.fp_detector:
            semantic_fps = self.fp_detector.detect_false_positives(
                baseline, revised, injection_manifest.get("errors", []), file_path
            )
            false_positives.extend(semantic_fps)

        return error_metrics, false_positives
    
    def _check_error_fixed(
        self,
        category: str,
        orig: str,
        mut: str,
        baseline: str,
        corrupted: str,
        revised: str,
        mutated_token: str = "",
    ) -> Tuple[bool, str]:
        """
        Check if a specific error was fixed.

        ``mutated_token`` is the optional token the injector recorded as the
        specific substring it introduced/changed (e.g. ``"--cores"`` for
        cli_unknown_flag). When provided for CLI categories, the fix check
        uses a whitespace-anchored token search instead of whole-line
        substring matching — this is robust to incidental edits the LLM may
        make to the surrounding line (e.g. collapsing a double space).

        Returns:
            Tuple of (is_fixed, status)
        """
        # Logic adapted from test_metrics.py
        if category == "typo":
            if orig and orig in revised:
                return True, "fixed_to_baseline"
            elif mut and mut in revised:
                return False, "unchanged"
            else:
                return True, "fixed_to_valid"
        
        elif category == "link":
            if orig and orig in revised:
                return True, "fixed_to_baseline"
            elif mut and mut in revised:
                return False, "unchanged"
            else:
                # Neither original nor mutated snippet present — link was rewritten
                return True, "rewritten"
        
        elif category == "duplicate":
            dup_before = corrupted.count(mut) if mut else 0
            dup_after = revised.count(mut) if mut else 0
            is_fixed = dup_after < dup_before
            return is_fixed, "fixed_to_valid" if is_fixed else "unchanged"
        
        elif category == "markdown_structure":
            issues_before = self._count_markdown_issues(corrupted)
            issues_after = self._count_markdown_issues(revised)
            is_fixed = issues_after < issues_before
            return is_fixed, "fixed_to_valid" if is_fixed else "unchanged"
        
        elif category in ("bio_term", "function"):
            if orig and orig in revised:
                return True, "fixed_to_baseline"
            elif mut and mut in revised:
                return False, "unchanged"
            else:
                return True, "fixed_to_valid"
        
        elif category == "list_structure":
            mal_before = len(re.findall(r"^[-*]\S", corrupted, flags=re.M))
            mal_after = len(re.findall(r"^[-*]\S", revised, flags=re.M))
            is_fixed = mal_after < mal_before
            return is_fixed, "fixed_to_valid" if is_fixed else "unchanged"
        
        elif category == "image_syntax":
            bad_before = len(re.findall(r"!\[[^\]]*\]\s+\(", corrupted))
            bad_after = len(re.findall(r"!\[[^\]]*\]\s+\(", revised))
            is_fixed = bad_after < bad_before
            return is_fixed, "fixed_to_valid" if is_fixed else "unchanged"
        
        elif category == "section_title":
            canonical_titles = {
                "## What is it?", "## What can it do?", "## Requirements",
                "## Install", "## Quick example", "## Learn more", "## License & Contact",
            }
            if mut and mut not in revised and any(t in revised for t in canonical_titles):
                return True, "fixed_to_valid"
            return False, "unchanged"
        
        elif category == "inline_code":
            raw = mut.strip('`') if mut else ""
            if not raw:
                return False, "unchanged"
            is_fixed = _naked_count(revised, raw) < _naked_count(corrupted, raw)
            return is_fixed, "fixed_to_valid" if is_fixed else "unchanged"
        
        elif category in ("emphasis", "code_lang_tag"):
            is_fixed = mut and mut not in revised
            return is_fixed, "fixed_to_valid" if is_fixed else "unchanged"
        
        elif category in ("number", "boolean", "param_name", "comment_typo", "species_name", "gene_case"):
            # For these categories: fixed if original restored OR mutated removed
            if orig and orig in revised:
                return True, "fixed_to_baseline"
            elif mut and mut in revised:
                return False, "unchanged"
            else:
                # Neither found = content rewritten = consider fixed
                return True, "fixed_to_valid"
        
        elif category == "table_alignment":
            var_before = self._table_variance(corrupted)
            var_after = self._table_variance(revised)
            is_fixed = var_after < var_before
            return is_fixed, "fixed_to_valid" if is_fixed else "unchanged"
        
        # CLI/Config categories injected by the new ``_inject_cli_consistency``
        # path: the manifest carries the specific token that was introduced
        # or changed by the mutation, so we can ask the precise question
        # "is that token still present, as a standalone whitespace-bounded
        # match, in the revised document?" — robust to incidental
        # whitespace/quoting edits on the same line.  When ``mutated_token``
        # is empty (legacy manifests), fall back to whole-line matching.
        elif category in {"cli_flag_typo", "cli_unknown_flag", "cli_program_rename"}:
            if mutated_token:
                pattern = re.compile(rf"(?<!\S){re.escape(mutated_token)}(?!\S)")
                is_fixed = pattern.search(revised) is None
            else:
                is_fixed = bool(mut) and mut not in revised
            return is_fixed, "fixed_to_valid" if is_fixed else "unchanged"

        # Biology-specific and CLI/CONFIG categories
        elif category in {
            "gene_symbol_case", "species_swap", "ref_genome_mismatch", "modality_confusion",
            "normalization_error", "umi_vs_read", "batch_effect", "qc_threshold", "file_format",
            "strandedness", "coordinates", "units_scale", "sample_type", "contamination",
            "param_name", "default_value", "path_hint",
        }:
            is_fixed = mut and mut not in revised
            return is_fixed, "fixed_to_valid" if is_fixed else "unchanged"
        
        # Default
        return False, "unchanged"
    
    def _count_markdown_issues(self, text: str) -> int:
        """Count markdown structural issues."""
        issues = 0
        issues += text.count("[![") - text.count("](")
        issues += text.count("[ ")
        issues += len(re.findall(r"^#[^#\s]", text, flags=re.M))
        return max(0, issues)
    
    def _table_variance(self, text: str) -> int:
        """Calculate table alignment variance."""
        rows = [ln for ln in text.splitlines() if '|' in ln]
        groups: List[List[str]] = []
        cur: List[str] = []
        for ln in rows:
            if '|' in ln:
                cur.append(ln)
            else:
                if len(cur) >= 2:
                    groups.append(cur)
                cur = []
        if len(cur) >= 2:
            groups.append(cur)
        vari = 0
        for g in groups:
            counts = [ln.count('|') for ln in g]
            vari += (max(counts) - min(counts))
        return vari
    
    def aggregate_results(
        self,
        all_error_metrics: List[ErrorMetrics],
        all_false_positives: List[FalsePositive],
        error_count: int,
        file_count: int
    ) -> BenchmarkResult:
        """
        Aggregate metrics from all files into a single BenchmarkResult.
        """
        result = BenchmarkResult(
            error_count=error_count,
            file_count=file_count,
        )
        
        # Count TP/FN from error metrics
        for em in all_error_metrics:
            if em.is_fixed:
                result.true_positives += 1
            else:
                result.false_negatives += 1
            
            # Per-category breakdown
            cat = em.category
            if cat not in result.per_category:
                result.per_category[cat] = {"tp": 0, "fn": 0}
            if em.is_fixed:
                result.per_category[cat]["tp"] += 1
            else:
                result.per_category[cat]["fn"] += 1
            
            # Per-file breakdown
            fp = em.file_path
            if fp not in result.per_file:
                result.per_file[fp] = {"tp": 0, "fn": 0, "fp": 0}
            if em.is_fixed:
                result.per_file[fp]["tp"] += 1
            else:
                result.per_file[fp]["fn"] += 1
            
            result.error_details.append(em)
        
        # Count FP from semantic detection
        result.false_positives = len(all_false_positives)
        result.fp_details = all_false_positives
        
        for fp in all_false_positives:
            if fp.file_path not in result.per_file:
                result.per_file[fp.file_path] = {"tp": 0, "fn": 0, "fp": 0}
            result.per_file[fp.file_path]["fp"] += 1
        
        # Compute derived metrics
        result.compute_derived_metrics()
        
        return result


def evaluate_benchmark(
    manifests: Dict[str, Dict[str, Any]],
    output_dir: str,
    llm: Optional[BaseChatOpenAI] = None,
    detect_semantic_fp: bool = True
) -> BenchmarkResult:
    """
    Evaluate a complete benchmark run.
    
    Args:
        manifests: Dict mapping file paths to their injection info
        output_dir: Directory containing the fixed files
        llm: LLM for semantic FP detection (optional)
        detect_semantic_fp: Whether to run semantic FP detection
        
    Returns:
        BenchmarkResult with all metrics
    """
    import os
    from bioguider.agents.agent_utils import read_file
    
    evaluator = BenchmarkEvaluator(llm)
    
    all_error_metrics: List[ErrorMetrics] = []
    all_false_positives: List[FalsePositive] = []
    total_errors = 0
    
    for rel_path, info in manifests.items():
        # Read fixed version
        fixed_path = os.path.join(output_dir, rel_path)
        if not os.path.exists(fixed_path):
            fixed_content = info["baseline_content"]
        else:
            fixed_content = read_file(fixed_path) or info["baseline_content"]
        
        # Evaluate this file
        error_metrics, false_positives = evaluator.evaluate_single_file(
            baseline=info["baseline_content"],
            corrupted=info["corrupted_content"],
            revised=fixed_content,
            injection_manifest=info["manifest"],
            file_path=rel_path,
            file_category=info["category"],
            detect_semantic_fp=detect_semantic_fp,
        )
        
        all_error_metrics.extend(error_metrics)
        all_false_positives.extend(false_positives)
        total_errors += len(info["manifest"].get("errors", []))
    
    # Aggregate results
    result = evaluator.aggregate_results(
        all_error_metrics,
        all_false_positives,
        error_count=total_errors,
        file_count=len(manifests),
    )
    
    return result

