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
    ],
    "cli_config": [
        "param_name",
        "default_value",
        "path_hint",
        "number",
        "boolean",
        "comment_typo",
    ],
}

# All error categories as a flat set
ALL_ERROR_CATEGORIES = frozenset(
    cat for cats in ERROR_CATEGORIES.values() for cat in cats
)


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
