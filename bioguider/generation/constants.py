"""
Shared constants for the document generation system.

This module centralizes all magic strings and values that were previously
scattered throughout the codebase.
"""

from __future__ import annotations

from enum import Enum
from typing import Set


class EvaluationScore(str, Enum):
    """Evaluation scores from the evaluation system."""

    EXCELLENT = "Excellent"
    GOOD = "Good"
    FAIR = "Fair"
    POOR = "Poor"

    @classmethod
    def needs_improvement(cls, score: str) -> bool:
        """Check if a score indicates the document needs improvement."""
        return score in (cls.FAIR.value, cls.POOR.value)

    @classmethod
    def all_values(cls) -> Set[str]:
        return {s.value for s in cls}


class FileCategory(str, Enum):
    """Categories of documentation files."""

    README = "readme"
    TUTORIAL = "tutorial"
    USERGUIDE = "userguide"
    INSTALLATION = "installation"

    @classmethod
    def all_values(cls) -> Set[str]:
        return {c.value for c in cls}


class ErrorCategory(str, Enum):
    """Categories of errors that can be injected/detected."""

    # Basic text errors
    TYPO = "typo"
    LINK = "link"
    DUPLICATE = "duplicate"

    # Markdown structure
    MARKDOWN_STRUCTURE = "markdown_structure"
    LIST_STRUCTURE = "list_structure"
    IMAGE_SYNTAX = "image_syntax"
    SECTION_TITLE = "section_title"
    INLINE_CODE = "inline_code"
    EMPHASIS = "emphasis"
    TABLE_ALIGNMENT = "table_alignment"
    CODE_LANG_TAG = "code_lang_tag"

    # Domain-specific (biology)
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

    # CLI/Config errors
    PARAM_NAME = "param_name"
    DEFAULT_VALUE = "default_value"
    PATH_HINT = "path_hint"
    CLI_FLAG_TYPO = "cli_flag_typo"
    CLI_UNKNOWN_FLAG = "cli_unknown_flag"
    CLI_PROGRAM_RENAME = "cli_program_rename"

    @classmethod
    def basic_categories(cls) -> Set[str]:
        """Categories for basic text/markdown errors."""
        return {
            cls.TYPO.value,
            cls.LINK.value,
            cls.DUPLICATE.value,
            cls.MARKDOWN_STRUCTURE.value,
            cls.LIST_STRUCTURE.value,
            cls.IMAGE_SYNTAX.value,
            cls.SECTION_TITLE.value,
            cls.INLINE_CODE.value,
            cls.EMPHASIS.value,
            cls.TABLE_ALIGNMENT.value,
            cls.CODE_LANG_TAG.value,
        }

    @classmethod
    def biology_categories(cls) -> Set[str]:
        """Categories for biology-specific errors."""
        return {
            cls.BIO_TERM.value,
            cls.FUNCTION.value,
            cls.GENE_SYMBOL_CASE.value,
            cls.SPECIES_SWAP.value,
            cls.REF_GENOME_MISMATCH.value,
            cls.MODALITY_CONFUSION.value,
            cls.NORMALIZATION_ERROR.value,
            cls.UMI_VS_READ.value,
            cls.BATCH_EFFECT.value,
            cls.QC_THRESHOLD.value,
            cls.FILE_FORMAT.value,
            cls.STRANDEDNESS.value,
            cls.COORDINATES.value,
            cls.UNITS_SCALE.value,
            cls.SAMPLE_TYPE.value,
            cls.CONTAMINATION.value,
        }

    @classmethod
    def cli_config_categories(cls) -> Set[str]:
        """Categories for CLI/config errors."""
        return {
            cls.PARAM_NAME.value,
            cls.DEFAULT_VALUE.value,
            cls.PATH_HINT.value,
            cls.CLI_FLAG_TYPO.value,
            cls.CLI_UNKNOWN_FLAG.value,
            cls.CLI_PROGRAM_RENAME.value,
        }

    @classmethod
    def all_values(cls) -> Set[str]:
        return {c.value for c in cls}


class FixStatus(str, Enum):
    """Status of an error fix attempt."""

    FIXED_TO_BASELINE = "fixed_to_baseline"  # Restored to exact original
    FIXED_TO_VALID = "fixed_to_valid"  # Fixed but not exactly to original
    UNCHANGED = "unchanged"  # Error still present
    WORSENED = "worsened"  # Made worse than corrupted


class EditType(str, Enum):
    """Types of document edits."""

    FULL_REPLACE = "full_replace"
    SECTION_INSERT = "section_insert"
    SECTION_REPLACE = "section_replace"
    APPEND = "append"
    PREPEND = "prepend"


# Canonical section titles for README structure validation
CANONICAL_README_TITLES = frozenset(
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

# Default file patterns for each category
DEFAULT_FILE_PATTERNS = {
    FileCategory.README.value: ["README.md", "README.rst", "README.txt"],
    FileCategory.TUTORIAL.value: ["*.Rmd", "vignettes/*.Rmd", "tutorials/*.md"],
    FileCategory.USERGUIDE.value: ["docs/*.md", "doc/*.md", "guide/*.md"],
    FileCategory.INSTALLATION.value: [
        "INSTALL.md",
        "INSTALLATION.md",
        "docs/installation.md",
    ],
}


# LLM generation limits
class GenerationLimits:
    """Limits for LLM generation."""

    MAX_TOKENS = 16384
    CONTENT_TRUNCATION = 30000
    CONTEXT_WINDOW = 8000
    CHUNK_SIZE = 6000
    MIN_SECTION_LENGTH = 50
    MAX_CONTINUATION_ATTEMPTS = 3


# Priority words for typo injection (common documentation terms)
PRIORITY_TYPO_WORDS = frozenset(
    {
        "installation",
        "successfully",
        "analysis",
        "documentation",
        "maintained",
        "example",
        "requirements",
        "license",
        "tutorials",
        "expression",
        "differential",
        "features",
        "cluster",
        "cells",
        "data",
        "sample",
        "marker",
        "gene",
        "function",
        "package",
        "method",
        "parameter",
        "variable",
        "object",
        "default",
        "optional",
        "required",
        "specify",
        "available",
        "different",
        "following",
        "particular",
        "similar",
        "significant",
        "corresponding",
        "additional",
        "individual",
    }
)
