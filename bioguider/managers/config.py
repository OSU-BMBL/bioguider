"""
Configuration dataclasses for BioGuider managers.

Centralizes configuration that was previously scattered across:
- Method parameters
- Hardcoded constants
- Module-level variables
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Callable, Any


@dataclass
class GenerationConfig:
    """Configuration for DocumentationGenerationManager."""

    # File processing limits
    max_files: Optional[int] = None
    target_files: Optional[List[str]] = None

    # Debug settings
    debug_output: bool = False
    debug_dir: str = "outputs/debug_generation"

    # Output settings
    output_dir: Optional[str] = None
    write_originals: bool = True

    # LLM settings
    clean_output: bool = True  # Run LLM cleaner on output


@dataclass
class TestConfig:
    """Configuration for GenerationTestManager and GenerationTestManagerV2."""

    # Error injection settings
    min_errors_per_category: int = 3

    # File selection
    max_files_per_category: int = 10
    include_readme: bool = True
    include_tutorials: bool = True
    include_userguides: bool = True
    include_installation: bool = True

    # Evaluation settings
    detect_semantic_fp: bool = False  # Disabled by default in test managers

    # Output settings
    save_original: bool = True
    save_corrupted: bool = True


@dataclass
class BenchmarkConfig:
    """Configuration for BenchmarkManager."""

    # Stress test settings
    stress_levels: List[int] = field(default_factory=lambda: [10, 20, 40, 60, 100])

    # Parallel execution
    max_workers: int = 4

    # File selection
    max_files_per_category: int = 10

    # Evaluation settings
    detect_semantic_fp: bool = True
    limit_generation_files: bool = True  # Only process injected files

    # Supported models for comparison
    comparison_models: List[str] = field(
        default_factory=lambda: ["bioguider", "gpt-4o", "claude-sonnet", "gemini"]
    )


@dataclass
class StepCallbackConfig:
    """Configuration for step callbacks."""

    callback: Optional[Callable[[str, str], None]] = None
    verbose: bool = True
    log_to_file: bool = False
    log_file_path: Optional[str] = None


# Default configurations
DEFAULT_GENERATION_CONFIG = GenerationConfig()
DEFAULT_TEST_CONFIG = TestConfig()
DEFAULT_BENCHMARK_CONFIG = BenchmarkConfig()


# Error categories organized by type
ERROR_CATEGORIES = {
    "text": [
        "typo",
        "link",
        "duplicate",
    ],
    "structure": [
        "markdown_structure",
        "list_structure",
        "table_alignment",
        "section_title",
        "image_syntax",
    ],
    "code": [
        "inline_code",
        "code_lang_tag",
        "emphasis",
    ],
    "biology": [
        "bio_term",
        "function",
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
        "accession_id_prefix",
    ],
    "cli_config": [
        "param_name",
        "default_value",
        "path_hint",
        "number",
        "boolean",
        "comment_typo",
    ],
    "prose_code_consistency": [
        "prose_code_pkg_version",
        "prose_code_stat_test",
        "prose_code_marker",
        "prose_code_param",
    ],
}

# All error categories as a flat set
ALL_ERROR_CATEGORIES = frozenset(
    cat for cats in ERROR_CATEGORIES.values() for cat in cats
)

# Categories that are injected for realism but excluded from F1 / precision / recall.
# Rationale: the BioGuider locator uses function names as anchors for finding
# documentation context. If we mutate a function name, the locator cannot
# re-establish context, so the error is structurally unfixable — scoring it
# would penalise the fixer for a design constraint, not a true failure.
# These errors still appear in per-category detail rows (pre-registered carve-out).
UNSCORABLE_CATEGORIES = frozenset({
    "function",
})

# Scorable categories are everything else (headline F1 denominator).
SCORABLE_CATEGORIES = ALL_ERROR_CATEGORIES - UNSCORABLE_CATEGORIES

# Category groups for paper-table breakdown (CONTENT vs HYGIENE).
# Rationale for tricky placements:
# - boolean -> HYGIENE: typically chunk options (echo=TRUE), display-only.
# - default_value -> CONTENT: wrong default is wrong analysis semantics.
# - path_hint -> HYGIENE: broken paths in prose are like broken links.
# - code_lang_tag -> HYGIENE: fence tag doesn't change execution.
# - All biology-group categories -> CONTENT: biological truth claims.
# - contamination -> CONTENT: wrong mechanism is misinformation.
CONTENT_CATEGORIES: frozenset = frozenset({
    # text/code fidelity
    "param_name", "gene_case", "bio_term", "species_name",
    "accession_id_prefix",
    "prose_code_pkg_version", "prose_code_stat_test",
    "prose_code_marker", "prose_code_param",
    "number",
    # biology group -- scientifically meaningful
    "gene_symbol_case", "species_swap", "ref_genome_mismatch",
    "modality_confusion", "normalization_error", "umi_vs_read",
    "batch_effect", "qc_threshold", "file_format", "strandedness",
    "coordinates", "units_scale", "sample_type", "contamination",
    # cli_config -- wrong default = wrong analysis
    "default_value",
})

HYGIENE_CATEGORIES: frozenset = frozenset({
    "typo", "comment_typo", "markdown_structure",
    "inline_code", "link", "duplicate",
    "boolean", "emphasis",
    # structure group -- pure markdown formatting
    "list_structure", "table_alignment", "section_title", "image_syntax",
    # code group -- language-tag on fence is presentational
    "code_lang_tag",
    # cli_config -- path hint is like broken link
    "path_hint",
})


def _validate_category_groups() -> None:
    """Invariant: CONTENT, HYGIENE, UNSCORABLE partition ALL_ERROR_CATEGORIES."""
    union = CONTENT_CATEGORIES | HYGIENE_CATEGORIES | UNSCORABLE_CATEGORIES
    all_cats = set(ALL_ERROR_CATEGORIES)
    overlap_ch = CONTENT_CATEGORIES & HYGIENE_CATEGORIES
    overlap_cu = CONTENT_CATEGORIES & UNSCORABLE_CATEGORIES
    overlap_hu = HYGIENE_CATEGORIES & UNSCORABLE_CATEGORIES
    if overlap_ch or overlap_cu or overlap_hu:
        raise ValueError(
            f"category groups overlap: {overlap_ch | overlap_cu | overlap_hu}"
        )
    missing = all_cats - union
    if missing:
        raise ValueError(f"categories not assigned to any group: {missing}")
    extra = union - all_cats
    if extra:
        raise ValueError(f"unknown categories in groups: {extra}")


_validate_category_groups()

# Total-error budget levels for the F1-vs-error-count gradient figure.
# Each entry is the TARGET total scorable errors injected across the repo
# at a single stress level. Consumed by BenchmarkManager.run_total_error_gradient.
TOTAL_ERROR_LEVELS = [50, 100, 200, 300]


def min_per_category_from_total(
    target_total_errors: int,
    n_files: int,
    n_categories: int,
) -> int:
    """Translate a total-errors budget into a per-category minimum.

    Injection hits N_files × N_categories category×file slots, but anchor
    requirements mean only a fraction actually produce errors. So we compute
    the per-slot target as an even spread and clamp to at least 1 (so every
    applicable category still fires).
    """
    import math
    denom = max(1, n_files * n_categories)
    return max(1, math.ceil(target_total_errors / denom))


def compute_scorable_breakdown(error_records, false_positives_total: int) -> dict:
    """Compute precision / recall / F1 / fix_rate with UNSCORABLE_CATEGORIES removed.

    Shared by ``bioguider.generation.unified_metrics.EvaluationResult`` and
    ``bioguider.generation.benchmark_metrics.BenchmarkResult`` so the scorable
    story stays identical across both evaluators. Either pass dataclass
    instances with ``.category`` / ``.is_fixed`` attrs or plain dicts with
    the same keys.

    Args:
        error_records: iterable of per-error records (ErrorEvaluation or
            ErrorMetrics instance or dict).
        false_positives_total: headline FP count; mirrored to the scorable
            bucket because FPs are category-agnostic (unintended changes to
            non-injected text).

    Returns:
        Dict with keys ``tp_scorable``, ``fn_scorable``, ``fp_scorable``,
        ``total_scorable``, ``precision_scorable``, ``recall_scorable``,
        ``f1_score_scorable``, ``fix_rate_scorable``.
    """
    def _get(rec, attr):
        if hasattr(rec, attr):
            return getattr(rec, attr)
        if isinstance(rec, dict):
            return rec.get(attr)
        return None

    tp_s = 0
    fn_s = 0
    for rec in error_records:
        if _get(rec, "category") in UNSCORABLE_CATEGORIES:
            continue
        if _get(rec, "is_fixed"):
            tp_s += 1
        else:
            fn_s += 1

    fp_s = false_positives_total
    total = tp_s + fn_s
    precision = tp_s / (tp_s + fp_s) if (tp_s + fp_s) > 0 else 0.0
    recall = tp_s / total if total > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    fix_rate = tp_s / total if total > 0 else 0.0
    return {
        "tp_scorable": tp_s,
        "fn_scorable": fn_s,
        "fp_scorable": fp_s,
        "total_scorable": total,
        "precision_scorable": precision,
        "recall_scorable": recall,
        "f1_score_scorable": f1,
        "fix_rate_scorable": fix_rate,
    }


def compute_group_breakdown(errors, fixed_ids, group_set):
    """Return (total_in_group, fixed_in_group) for errors whose category in group_set.

    Accepts iterables of dataclass instances or dicts (shape mirrors
    ``compute_scorable_breakdown``).
    """
    def _get(rec, attr):
        if hasattr(rec, attr):
            return getattr(rec, attr)
        if isinstance(rec, dict):
            return rec.get(attr)
        return None

    fixed_ids_set = set(fixed_ids) if fixed_ids is not None else set()
    total = 0
    fixed = 0
    for e in errors:
        if _get(e, "category") not in group_set:
            continue
        total += 1
        if _get(e, "id") in fixed_ids_set:
            fixed += 1
    return total, fixed


# File category definitions
FILE_CATEGORIES = {
    "readme": {
        "patterns": ["README.md", "README.rst", "README.txt", "readme.md"],
        "description": "Repository README files",
    },
    "tutorial": {
        "patterns": ["vignettes/*.Rmd", "tutorials/*.md"],
        "description": "Tutorial and vignette files",
    },
    "userguide": {
        "patterns": ["docs/*.md", "USERGUIDE.md", "GUIDE.md"],
        "description": "User guide and documentation files",
    },
    "installation": {
        "patterns": ["install*.md", "INSTALL*.md", "installation*.md"],
        "description": "Installation documentation",
    },
}


def merge_configs(base: Any, override: Any) -> Any:
    """
    Merge two configuration objects, with override taking precedence.

    Args:
        base: Base configuration dataclass
        override: Override configuration (can be partial dict or dataclass)

    Returns:
        New configuration with merged values
    """
    if override is None:
        return base

    if isinstance(override, dict):
        # Create new instance with base values, then override
        base_dict = {k: v for k, v in base.__dict__.items()}
        base_dict.update({k: v for k, v in override.items() if v is not None})
        return type(base)(**base_dict)
    else:
        # Both are dataclasses
        base_dict = {k: v for k, v in base.__dict__.items()}
        override_dict = {k: v for k, v in override.__dict__.items() if v is not None}
        base_dict.update(override_dict)
        return type(base)(**base_dict)
