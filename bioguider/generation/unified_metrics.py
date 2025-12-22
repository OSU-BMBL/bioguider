"""
Unified metrics evaluation for error injection testing and benchmarking.

This module consolidates logic from:
- test_metrics.py (legacy evaluation for GenerationTestManager)
- benchmark_metrics.py (F-score evaluation for BenchmarkManager)

It provides a single, consistent API for evaluating documentation fixes.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher, unified_diff
from enum import Enum
from typing import Dict, Any, List, Tuple, Optional

try:
    from langchain_openai.chat_models.base import BaseChatOpenAI
except ImportError:
    BaseChatOpenAI = Any  # type: ignore


class FixStatus(str, Enum):
    """Status of an error fix attempt."""

    FIXED_TO_BASELINE = "fixed_to_baseline"  # Restored to original
    FIXED_TO_VALID = "fixed_to_valid"  # Fixed to valid alternative
    UNCHANGED = "unchanged"  # Error not fixed
    WORSENED = "worsened"  # Made worse (rare)


class ErrorCategory(str, Enum):
    """Categories of injected errors."""

    # Text errors
    TYPO = "typo"
    LINK = "link"
    DUPLICATE = "duplicate"

    # Structure errors
    MARKDOWN_STRUCTURE = "markdown_structure"
    LIST_STRUCTURE = "list_structure"
    TABLE_ALIGNMENT = "table_alignment"
    SECTION_TITLE = "section_title"

    # Code errors
    INLINE_CODE = "inline_code"
    CODE_LANG_TAG = "code_lang_tag"
    EMPHASIS = "emphasis"
    IMAGE_SYNTAX = "image_syntax"

    # Biology-specific errors
    BIO_TERM = "bio_term"
    FUNCTION = "function"
    GENE_SYMBOL_CASE = "gene_symbol_case"
    SPECIES_SWAP = "species_swap"
    REF_GENOME_MISMATCH = "ref_genome_mismatch"
    MODALITY_CONFUSION = "modality_confusion"
    NORMALIZATION_ERROR = "normalization_error"
    UMI_VS_READ = "umi_vs_read"
    BATCH_EFFECT = "batch_effect"
    QC_THRESHOLD = "qc_threshold"
    FILE_FORMAT = "file_format"
    STRANDEDNESS = "strandedness"
    COORDINATES = "coordinates"
    UNITS_SCALE = "units_scale"
    SAMPLE_TYPE = "sample_type"
    CONTAMINATION = "contamination"
    SPECIES_NAME = "species_name"
    GENE_CASE = "gene_case"

    # CLI/Config errors
    PARAM_NAME = "param_name"
    DEFAULT_VALUE = "default_value"
    PATH_HINT = "path_hint"
    NUMBER = "number"
    BOOLEAN = "boolean"
    COMMENT_TYPO = "comment_typo"


@dataclass
class ErrorEvaluation:
    """Evaluation result for a single error."""

    error_id: str
    category: str
    file_path: str
    status: FixStatus
    is_fixed: bool
    original_snippet: str = ""
    mutated_snippet: str = ""
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.error_id,
            "category": self.category,
            "file_path": self.file_path,
            "status": self.status.value,
            "is_fixed": self.is_fixed,
            "notes": self.notes,
        }


@dataclass
class FalsePositive:
    """Represents a detected false positive (harmful unintended change)."""

    file_path: str
    change_description: str
    severity: str  # "harmful", "neutral", "beneficial"
    original_text: str = ""
    changed_text: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "file_path": self.file_path,
            "change_description": self.change_description,
            "severity": self.severity,
        }


@dataclass
class CategoryMetrics:
    """Metrics for a single error category."""

    total: int = 0
    fixed_to_baseline: int = 0
    fixed_to_valid: int = 0
    unchanged: int = 0
    worsened: int = 0

    @property
    def fixed(self) -> int:
        return self.fixed_to_baseline + self.fixed_to_valid

    @property
    def fix_rate(self) -> float:
        return self.fixed / self.total if self.total > 0 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total": self.total,
            "fixed_to_baseline": self.fixed_to_baseline,
            "fixed_to_valid": self.fixed_to_valid,
            "unchanged": self.unchanged,
            "worsened": self.worsened,
            "fix_rate": round(self.fix_rate, 4),
        }


@dataclass
class FileMetrics:
    """Metrics for a single file."""

    file_path: str
    category: str = ""
    true_positives: int = 0
    false_negatives: int = 0
    false_positives: int = 0

    @property
    def total_errors(self) -> int:
        return self.true_positives + self.false_negatives

    def to_dict(self) -> Dict[str, Any]:
        return {
            "file_path": self.file_path,
            "category": self.category,
            "tp": self.true_positives,
            "fn": self.false_negatives,
            "fp": self.false_positives,
        }


@dataclass
class EvaluationResult:
    """Complete evaluation result for a benchmark run."""

    # Counts
    total_files: int = 0
    total_errors: int = 0

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
    success_rate: float = 0.0  # Legacy compatibility

    # Detailed breakdowns
    per_category: Dict[str, CategoryMetrics] = field(default_factory=dict)
    per_file: Dict[str, FileMetrics] = field(default_factory=dict)
    error_evaluations: List[ErrorEvaluation] = field(default_factory=list)
    false_positive_details: List[FalsePositive] = field(default_factory=list)

    # Global metrics
    markdown_validity_delta: int = 0

    def compute_metrics(self):
        """Compute all derived metrics from raw counts."""
        # Precision = TP / (TP + FP)
        if self.true_positives + self.false_positives > 0:
            self.precision = self.true_positives / (
                self.true_positives + self.false_positives
            )
        else:
            self.precision = 0.0

        # Recall = TP / (TP + FN)
        if self.true_positives + self.false_negatives > 0:
            self.recall = self.true_positives / (
                self.true_positives + self.false_negatives
            )
        else:
            self.recall = 0.0

        # F1 = 2 * (precision * recall) / (precision + recall)
        if self.precision + self.recall > 0:
            self.f1_score = (
                2 * (self.precision * self.recall) / (self.precision + self.recall)
            )
        else:
            self.f1_score = 0.0

        # Fix rate = TP / (TP + FN)
        total_errors = self.true_positives + self.false_negatives
        if total_errors > 0:
            self.fix_rate = self.true_positives / total_errors
            self.success_rate = round(self.fix_rate * 100, 2)
        else:
            self.fix_rate = 0.0
            self.success_rate = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "total_files": self.total_files,
            "total_errors": self.total_errors,
            "true_positives": self.true_positives,
            "false_negatives": self.false_negatives,
            "false_positives": self.false_positives,
            "true_negatives": self.true_negatives,
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1_score": round(self.f1_score, 4),
            "fix_rate": round(self.fix_rate, 4),
            "success_rate": self.success_rate,
            "per_category": {k: v.to_dict() for k, v in self.per_category.items()},
            "per_file": {k: v.to_dict() for k, v in self.per_file.items()},
            "error_evaluations": [e.to_dict() for e in self.error_evaluations],
            "false_positive_details": [
                fp.to_dict() for fp in self.false_positive_details
            ],
            "markdown_validity_delta": self.markdown_validity_delta,
        }

    def to_legacy_format(self) -> Dict[str, Any]:
        """
        Convert to legacy format for backward compatibility with test_metrics.

        Returns format expected by GenerationTestManager.
        """
        per_error = [
            {
                "id": e.error_id,
                "category": e.category,
                "status": e.status.value,
                "before": e.mutated_snippet,
                "after_contains_original": e.status == FixStatus.FIXED_TO_BASELINE,
                "notes": e.notes,
            }
            for e in self.error_evaluations
        ]

        per_category = {
            cat: {
                "total": m.total,
                "fixed_to_baseline": m.fixed_to_baseline,
                "fixed_to_valid": m.fixed_to_valid,
                "unchanged": m.unchanged,
                "worsened": m.worsened,
            }
            for cat, m in self.per_category.items()
        }

        totals = {
            "total_errors": self.total_errors,
            "fixed_to_baseline": sum(
                m.fixed_to_baseline for m in self.per_category.values()
            ),
            "fixed_to_valid": sum(m.fixed_to_valid for m in self.per_category.values()),
            "unchanged": self.false_negatives,
            "worsened": 0,
        }

        return {
            "per_error": per_error,
            "per_category": per_category,
            "global": {"markdown_validity_delta": self.markdown_validity_delta},
            "summary": {
                "totals": totals,
                "success_rate": self.success_rate,
            },
        }


class ErrorChecker:
    """
    Checks whether individual errors have been fixed.

    Consolidates the error-checking logic from test_metrics and benchmark_metrics.
    """

    # Canonical section titles for validation
    CANONICAL_TITLES = frozenset(
        {
            "## What is it?",
            "## What can it do?",
            "## Requirements",
            "## Install",
            "## Quick example",
            "## Learn more",
            "## License & Contact",
        }
    )

    def check_error_fixed(
        self,
        category: str,
        original: str,
        mutated: str,
        baseline: str,
        corrupted: str,
        revised: str,
    ) -> Tuple[bool, FixStatus]:
        """
        Check if a specific error was fixed.

        Args:
            category: Error category
            original: Original (correct) snippet
            mutated: Mutated (error) snippet
            baseline: Original full document
            corrupted: Corrupted full document
            revised: Revised/fixed full document

        Returns:
            Tuple of (is_fixed, status)
        """
        # Dispatch to category-specific checker
        checker = self._get_checker(category)
        return checker(original, mutated, baseline, corrupted, revised)

    def _get_checker(self, category: str):
        """Get the appropriate checker function for a category."""
        checkers = {
            "typo": self._check_typo,
            "link": self._check_link,
            "duplicate": self._check_duplicate,
            "markdown_structure": self._check_markdown_structure,
            "list_structure": self._check_list_structure,
            "image_syntax": self._check_image_syntax,
            "section_title": self._check_section_title,
            "inline_code": self._check_inline_code,
            "emphasis": self._check_emphasis,
            "table_alignment": self._check_table_alignment,
            "code_lang_tag": self._check_code_lang_tag,
            "bio_term": self._check_text_restoration,
            "function": self._check_text_restoration,
        }

        # Biology-specific categories
        bio_categories = {
            "gene_symbol_case",
            "species_swap",
            "ref_genome_mismatch",
            "modality_confusion",
            "normalization_error",
            "umi_vs_read",
            "batch_effect",
            "qc_threshold",
            "file_format",
            "strandedness",
            "coordinates",
            "units_scale",
            "sample_type",
            "contamination",
            "species_name",
            "gene_case",
        }

        # CLI/Config categories
        cli_categories = {
            "param_name",
            "default_value",
            "path_hint",
            "number",
            "boolean",
            "comment_typo",
        }

        if category in checkers:
            return checkers[category]
        elif category in bio_categories:
            return self._check_mutated_removed
        elif category in cli_categories:
            return self._check_text_restoration
        else:
            return self._check_default

    def _check_typo(
        self, orig, mut, baseline, corrupted, revised
    ) -> Tuple[bool, FixStatus]:
        if orig and orig in revised:
            return True, FixStatus.FIXED_TO_BASELINE
        elif mut and mut in revised:
            return False, FixStatus.UNCHANGED
        else:
            return True, FixStatus.FIXED_TO_VALID

    def _check_link(
        self, orig, mut, baseline, corrupted, revised
    ) -> Tuple[bool, FixStatus]:
        # Check if any well-formed link exists
        wellformed = re.search(r"\[[^\]]+\]\([^\s)]+\)", revised) is not None
        return (
            wellformed,
            FixStatus.FIXED_TO_VALID if wellformed else FixStatus.UNCHANGED,
        )

    def _check_duplicate(
        self, orig, mut, baseline, corrupted, revised
    ) -> Tuple[bool, FixStatus]:
        if not mut:
            return False, FixStatus.UNCHANGED
        dup_before = corrupted.count(mut)
        dup_after = revised.count(mut)
        is_fixed = dup_after < dup_before
        return is_fixed, FixStatus.FIXED_TO_VALID if is_fixed else FixStatus.UNCHANGED

    def _check_markdown_structure(
        self, orig, mut, baseline, corrupted, revised
    ) -> Tuple[bool, FixStatus]:
        issues_before = self._count_markdown_issues(corrupted)
        issues_after = self._count_markdown_issues(revised)
        is_fixed = issues_after < issues_before
        return is_fixed, FixStatus.FIXED_TO_VALID if is_fixed else FixStatus.UNCHANGED

    def _check_list_structure(
        self, orig, mut, baseline, corrupted, revised
    ) -> Tuple[bool, FixStatus]:
        mal_before = len(re.findall(r"^[-*]\S", corrupted, flags=re.M))
        mal_after = len(re.findall(r"^[-*]\S", revised, flags=re.M))
        is_fixed = mal_after < mal_before
        return is_fixed, FixStatus.FIXED_TO_VALID if is_fixed else FixStatus.UNCHANGED

    def _check_image_syntax(
        self, orig, mut, baseline, corrupted, revised
    ) -> Tuple[bool, FixStatus]:
        bad_before = len(re.findall(r"!\[[^\]]*\]\s+\(", corrupted))
        bad_after = len(re.findall(r"!\[[^\]]*\]\s+\(", revised))
        is_fixed = bad_after < bad_before
        return is_fixed, FixStatus.FIXED_TO_VALID if is_fixed else FixStatus.UNCHANGED

    def _check_section_title(
        self, orig, mut, baseline, corrupted, revised
    ) -> Tuple[bool, FixStatus]:
        if (
            mut
            and mut not in revised
            and any(t in revised for t in self.CANONICAL_TITLES)
        ):
            return True, FixStatus.FIXED_TO_VALID
        return False, FixStatus.UNCHANGED

    def _check_inline_code(
        self, orig, mut, baseline, corrupted, revised
    ) -> Tuple[bool, FixStatus]:
        raw = mut.strip("`") if mut else ""
        rewrapped = f"`{raw}`" if raw else ""
        if raw and rewrapped and rewrapped in revised and mut not in revised:
            return True, FixStatus.FIXED_TO_VALID
        return False, FixStatus.UNCHANGED

    def _check_emphasis(
        self, orig, mut, baseline, corrupted, revised
    ) -> Tuple[bool, FixStatus]:
        is_fixed = bool(mut and mut not in revised)
        return is_fixed, FixStatus.FIXED_TO_VALID if is_fixed else FixStatus.UNCHANGED

    def _check_table_alignment(
        self, orig, mut, baseline, corrupted, revised
    ) -> Tuple[bool, FixStatus]:
        var_before = self._table_variance(corrupted)
        var_after = self._table_variance(revised)
        is_fixed = var_after < var_before
        return is_fixed, FixStatus.FIXED_TO_VALID if is_fixed else FixStatus.UNCHANGED

    def _check_code_lang_tag(
        self, orig, mut, baseline, corrupted, revised
    ) -> Tuple[bool, FixStatus]:
        is_fixed = bool(mut and mut not in revised)
        return is_fixed, FixStatus.FIXED_TO_VALID if is_fixed else FixStatus.UNCHANGED

    def _check_text_restoration(
        self, orig, mut, baseline, corrupted, revised
    ) -> Tuple[bool, FixStatus]:
        """Check if original text was restored or mutated text was corrected."""
        if orig and orig in revised:
            return True, FixStatus.FIXED_TO_BASELINE
        elif mut and mut in revised:
            return False, FixStatus.UNCHANGED
        else:
            # Neither found = content rewritten = consider fixed
            return True, FixStatus.FIXED_TO_VALID

    def _check_mutated_removed(
        self, orig, mut, baseline, corrupted, revised
    ) -> Tuple[bool, FixStatus]:
        """Check if the mutated content was removed."""
        is_fixed = bool(mut and mut not in revised)
        return is_fixed, FixStatus.FIXED_TO_VALID if is_fixed else FixStatus.UNCHANGED

    def _check_default(
        self, orig, mut, baseline, corrupted, revised
    ) -> Tuple[bool, FixStatus]:
        """Default checker - unchanged."""
        return False, FixStatus.UNCHANGED

    @staticmethod
    def _count_markdown_issues(text: str) -> int:
        """Count markdown structural issues."""
        issues = 0
        issues += text.count("[![") - text.count("](")  # Unbalanced badge syntax
        issues += text.count("[ ")  # Bad link spacing
        issues += len(re.findall(r"^#[^#\s]", text, flags=re.M))  # Malformed header
        return max(0, issues)

    @staticmethod
    def _table_variance(text: str) -> int:
        """Calculate table alignment variance."""
        rows = [ln for ln in text.splitlines() if "|" in ln]
        groups: List[List[str]] = []
        cur: List[str] = []
        for ln in rows:
            if "|" in ln:
                cur.append(ln)
            else:
                if len(cur) >= 2:
                    groups.append(cur)
                cur = []
        if len(cur) >= 2:
            groups.append(cur)

        variance = 0
        for group in groups:
            counts = [ln.count("|") for ln in group]
            variance += max(counts) - min(counts)
        return variance


class SemanticFPDetector:
    """
    Detects false positives using LLM semantic analysis.

    A false positive is a harmful unintended change made by the fixer
    that was not related to fixing an injected error.
    """

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

    def __init__(self, llm: Optional[Any] = None):
        self.llm = llm

    def detect(
        self,
        baseline: str,
        revised: str,
        injected_errors: List[Dict[str, Any]],
        file_path: str,
    ) -> List[FalsePositive]:
        """
        Detect harmful unintended changes (false positives).

        Args:
            baseline: Original correct content
            revised: Content after AI fixes
            injected_errors: List of errors that were intentionally injected
            file_path: Path to the file being analyzed

        Returns:
            List of detected false positives (harmful changes only)
        """
        if self.llm is None:
            return []

        # Generate unified diff
        baseline_lines = baseline.splitlines(keepends=True)
        revised_lines = revised.splitlines(keepends=True)
        diff_lines = list(
            unified_diff(
                baseline_lines,
                revised_lines,
                fromfile="baseline",
                tofile="revised",
                lineterm="",
            )
        )
        diff_text = "".join(diff_lines)

        if not diff_text.strip():
            return []

        # Format injected errors for prompt
        error_descriptions = []
        for err in injected_errors:
            error_descriptions.append(
                f"- Category: {err.get('category', 'unknown')}\n"
                f"  Original: {str(err.get('original_snippet', 'N/A'))[:100]}\n"
                f"  Mutated: {str(err.get('mutated_snippet', 'N/A'))[:100]}"
            )
        errors_text = "\n".join(error_descriptions) if error_descriptions else "None"

        # Build prompt
        prompt = self.SEMANTIC_FP_PROMPT.format(
            injected_errors=errors_text,
            diff=diff_text[:8000],  # Limit diff size
        )

        try:
            from bioguider.agents.common_conversation import CommonConversation

            conv = CommonConversation(self.llm)
            output, _ = conv.generate(
                system_prompt=prompt,
                instruction_prompt="Analyze the changes and return the JSON.",
            )

            result = self._parse_json_output(output)

            # Extract harmful changes as false positives
            false_positives = []
            for change in result.get("unintended_changes", []):
                if change.get("severity") == "harmful":
                    false_positives.append(
                        FalsePositive(
                            file_path=file_path,
                            change_description=change.get(
                                "description", "Unknown change"
                            ),
                            severity="harmful",
                            original_text=change.get("original_text", ""),
                            changed_text=change.get("changed_text", ""),
                        )
                    )

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

        # Strategy 2: Extract JSON block from markdown
        json_pattern = r"```(?:json)?\s*(\{.*?\})\s*```"
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
                    return json.loads(output[start : end + 1])
                except json.JSONDecodeError:
                    pass

        # Fallback
        return {
            "unintended_changes": [],
            "summary": {"harmful_count": 0, "neutral_count": 0, "beneficial_count": 0},
        }


class UnifiedMetricsEvaluator:
    """
    Unified evaluator for error injection testing.

    Consolidates logic from:
    - test_metrics.evaluate_fixes()
    - benchmark_metrics.BenchmarkEvaluator
    """

    def __init__(self, llm: Optional[Any] = None, detect_fp: bool = True):
        """
        Initialize the evaluator.

        Args:
            llm: Optional LLM for semantic false positive detection
            detect_fp: Whether to detect false positives
        """
        self.error_checker = ErrorChecker()
        self.fp_detector = SemanticFPDetector(llm) if detect_fp else None
        self.detect_fp = detect_fp and llm is not None

    def evaluate_single_file(
        self,
        baseline: str,
        corrupted: str,
        revised: str,
        injection_manifest: Dict[str, Any],
        file_path: str = "",
        file_category: str = "",
    ) -> Tuple[List[ErrorEvaluation], List[FalsePositive]]:
        """
        Evaluate fixes for a single file.

        Args:
            baseline: Original correct content
            corrupted: Content with injected errors
            revised: Content after fixing attempt
            injection_manifest: Manifest of injected errors
            file_path: Path to the file
            file_category: Category of the file (readme, tutorial, etc.)

        Returns:
            Tuple of (error_evaluations, false_positives)
        """
        evaluations = []

        for error in injection_manifest.get("errors", []):
            error_id = error.get("id", "unknown")
            category = error.get("category", "unknown")
            original = error.get("original_snippet", "")
            mutated = error.get("mutated_snippet", "")

            is_fixed, status = self.error_checker.check_error_fixed(
                category, original, mutated, baseline, corrupted, revised
            )

            evaluations.append(
                ErrorEvaluation(
                    error_id=error_id,
                    category=category,
                    file_path=file_path,
                    status=status,
                    is_fixed=is_fixed,
                    original_snippet=original,
                    mutated_snippet=mutated,
                )
            )

        # Detect false positives if enabled
        false_positives = []
        if self.detect_fp and self.fp_detector:
            false_positives = self.fp_detector.detect(
                baseline, revised, injection_manifest.get("errors", []), file_path
            )

        return evaluations, false_positives

    def evaluate_multiple_files(
        self,
        manifests: Dict[str, Dict[str, Any]],
        output_dir: str,
    ) -> EvaluationResult:
        """
        Evaluate fixes across multiple files.

        Args:
            manifests: Dict mapping file paths to their injection info
            output_dir: Directory containing the fixed files

        Returns:
            Complete EvaluationResult
        """
        import os
        from bioguider.agents.agent_utils import read_file

        result = EvaluationResult(
            total_files=len(manifests),
        )

        all_evaluations = []
        all_false_positives = []

        for rel_path, info in manifests.items():
            # Read fixed version
            fixed_path = os.path.join(output_dir, rel_path)
            if os.path.exists(fixed_path):
                fixed_content = read_file(fixed_path) or info.get(
                    "baseline_content", ""
                )
            else:
                fixed_content = info.get("baseline_content", "")

            # Evaluate this file
            file_evals, file_fps = self.evaluate_single_file(
                baseline=info.get("baseline_content", ""),
                corrupted=info.get("corrupted_content", ""),
                revised=fixed_content,
                injection_manifest=info.get("manifest", {}),
                file_path=rel_path,
                file_category=info.get("category", ""),
            )

            all_evaluations.extend(file_evals)
            all_false_positives.extend(file_fps)

            # Track per-file metrics
            file_metrics = FileMetrics(
                file_path=rel_path, category=info.get("category", "")
            )
            for eval in file_evals:
                if eval.is_fixed:
                    file_metrics.true_positives += 1
                else:
                    file_metrics.false_negatives += 1
            file_metrics.false_positives = len(
                [fp for fp in file_fps if fp.file_path == rel_path]
            )
            result.per_file[rel_path] = file_metrics

        # Aggregate results
        result = self._aggregate_results(all_evaluations, all_false_positives, result)

        return result

    def _aggregate_results(
        self,
        evaluations: List[ErrorEvaluation],
        false_positives: List[FalsePositive],
        result: EvaluationResult,
    ) -> EvaluationResult:
        """Aggregate individual evaluations into final result."""
        result.total_errors = len(evaluations)
        result.error_evaluations = evaluations
        result.false_positive_details = false_positives

        for eval in evaluations:
            if eval.is_fixed:
                result.true_positives += 1
            else:
                result.false_negatives += 1

            # Per-category tracking
            cat = eval.category
            if cat not in result.per_category:
                result.per_category[cat] = CategoryMetrics()

            result.per_category[cat].total += 1
            if eval.status == FixStatus.FIXED_TO_BASELINE:
                result.per_category[cat].fixed_to_baseline += 1
            elif eval.status == FixStatus.FIXED_TO_VALID:
                result.per_category[cat].fixed_to_valid += 1
            elif eval.status == FixStatus.UNCHANGED:
                result.per_category[cat].unchanged += 1
            elif eval.status == FixStatus.WORSENED:
                result.per_category[cat].worsened += 1

        result.false_positives = len(false_positives)

        # Compute derived metrics
        result.compute_metrics()

        return result


# Convenience function for backward compatibility with test_metrics
def evaluate_fixes(
    baseline: str,
    corrupted: str,
    revised: str,
    injection_manifest: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Backward-compatible wrapper for legacy code.

    Use UnifiedMetricsEvaluator.evaluate_single_file() for new code.
    """
    evaluator = UnifiedMetricsEvaluator(llm=None, detect_fp=False)
    evals, _ = evaluator.evaluate_single_file(
        baseline, corrupted, revised, injection_manifest
    )

    # Build legacy result format
    result = EvaluationResult(total_errors=len(evals))
    result = evaluator._aggregate_results(evals, [], result)

    # Compute markdown validity delta
    result.markdown_validity_delta = ErrorChecker._count_markdown_issues(
        corrupted
    ) - ErrorChecker._count_markdown_issues(revised)

    return result.to_legacy_format()


# Convenience function for backward compatibility with benchmark_metrics
def evaluate_benchmark(
    manifests: Dict[str, Dict[str, Any]],
    output_dir: str,
    llm: Optional[Any] = None,
    detect_semantic_fp: bool = True,
) -> EvaluationResult:
    """
    Backward-compatible wrapper for legacy code.

    Use UnifiedMetricsEvaluator.evaluate_multiple_files() for new code.
    """
    evaluator = UnifiedMetricsEvaluator(llm=llm, detect_fp=detect_semantic_fp)
    return evaluator.evaluate_multiple_files(manifests, output_dir)
