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
            
            # Determine if error was fixed
            is_fixed, status = self._check_error_fixed(
                category, orig, mut, baseline, corrupted, revised
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
        
        # Detect false positives if LLM available and enabled
        false_positives = []
        if detect_semantic_fp and self.fp_detector:
            false_positives = self.fp_detector.detect_false_positives(
                baseline, revised, injection_manifest.get("errors", []), file_path
            )
        
        return error_metrics, false_positives
    
    def _check_error_fixed(
        self,
        category: str,
        orig: str,
        mut: str,
        baseline: str,
        corrupted: str,
        revised: str
    ) -> Tuple[bool, str]:
        """
        Check if a specific error was fixed.
        
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
            wellformed = re.search(r"\[[^\]]+\]\([^\s)]+\)", revised) is not None
            return wellformed, "fixed_to_valid" if wellformed else "unchanged"
        
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
            rewrapped = f"`{raw}`" if raw else ""
            if raw and rewrapped and rewrapped in revised and mut not in revised:
                return True, "fixed_to_valid"
            return False, "unchanged"
        
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
        
        # Biology-specific and CLI/CONFIG categories
        elif category in {
            "gene_symbol_case", "species_swap", "ref_genome_mismatch", "modality_confusion",
            "normalization_error", "umi_vs_read", "batch_effect", "qc_threshold", "file_format",
            "strandedness", "coordinates", "units_scale", "sample_type", "contamination",
            "param_name", "default_value", "path_hint"
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

