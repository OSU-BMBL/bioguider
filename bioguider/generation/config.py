"""
Configuration dataclasses for the LLM content generation components.

This module provides configuration for:
- LLM generation parameters (tokens, temperature, etc.)
- Content processing limits
- Error injection settings
- Metrics evaluation settings

Note: For manager-level configuration (output dirs, file selection),
see bioguider.managers.config.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Optional, Dict

from .constants import FileCategory, GenerationLimits


@dataclass
class LLMGenerationConfig:
    """
    Configuration for LLM content generation.

    This configures the LLMContentGenerator and related components.
    For high-level pipeline config, see managers.config.GenerationConfig.
    """

    # LLM settings
    model_name: str = "gpt-4o"
    max_tokens: int = GenerationLimits.MAX_TOKENS
    temperature: float = 0.7

    # Content limits
    content_truncation_threshold: int = GenerationLimits.CONTENT_TRUNCATION
    context_window_size: int = GenerationLimits.CONTEXT_WINDOW
    chunk_size: int = GenerationLimits.CHUNK_SIZE
    max_continuation_attempts: int = GenerationLimits.MAX_CONTINUATION_ATTEMPTS

    # Processing options
    clean_output: bool = True  # Run LLM cleaner on output
    preserve_code_blocks: bool = True  # Extra validation for code blocks

    # Debug settings
    debug_enabled: bool = False
    debug_dir: str = "outputs/debug_generation"

    @classmethod
    def from_environment(cls) -> "LLMGenerationConfig":
        """Create config from environment variables."""
        return cls(
            model_name=os.environ.get("OPENAI_MODEL", "gpt-4o"),
            debug_enabled=os.environ.get("BIOGUIDER_DEBUG", "").lower() == "true",
            debug_dir=os.environ.get("BIOGUIDER_DEBUG_DIR", "outputs/debug_generation"),
        )


@dataclass
class InjectionConfig:
    """Configuration for error injection."""

    min_per_category: int = 3
    max_words: int = 450

    # Preserve these keywords from corruption
    preserve_keywords: Optional[List[str]] = None

    # Project-specific terms to target for corruption
    project_terms: Optional[List[str]] = None

    # Category weights (higher = more errors of this type)
    category_weights: Dict[str, float] = field(default_factory=dict)

    # Similarity threshold for validation
    min_similarity: float = 0.75


@dataclass
class MetricsConfig:
    """Configuration for metrics evaluation."""

    # F-score calculation
    compute_f_score: bool = True

    # Semantic false positive detection
    detect_semantic_fp: bool = False
    semantic_fp_model: str = "gpt-4o"

    # Thresholds
    similarity_threshold: float = 0.8

    # Output options
    detailed_per_error: bool = True
    aggregate_by_category: bool = True


# Re-export commonly used configs for convenience
# Manager-level configs come from bioguider.managers.config
__all__ = [
    "LLMGenerationConfig",
    "InjectionConfig",
    "MetricsConfig",
]
