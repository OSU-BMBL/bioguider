"""Unit tests for generation config and constants."""

import pytest
from bioguider.generation.constants import (
    EvaluationScore,
    FileCategory,
    ErrorCategory,
    FixStatus,
    EditType,
    CANONICAL_README_TITLES,
    GenerationLimits,
)
from bioguider.generation.config import (
    LLMGenerationConfig,
    InjectionConfig,
    MetricsConfig,
)
from bioguider.managers.config import BenchmarkConfig


class TestEvaluationScore:
    def test_needs_improvement_poor(self):
        assert EvaluationScore.needs_improvement("Poor") is True

    def test_needs_improvement_fair(self):
        assert EvaluationScore.needs_improvement("Fair") is True

    def test_needs_improvement_good(self):
        assert EvaluationScore.needs_improvement("Good") is False

    def test_needs_improvement_excellent(self):
        assert EvaluationScore.needs_improvement("Excellent") is False

    def test_all_values(self):
        values = EvaluationScore.all_values()
        assert "Excellent" in values
        assert "Good" in values
        assert "Fair" in values
        assert "Poor" in values
        assert len(values) == 4


class TestFileCategory:
    def test_all_categories_exist(self):
        assert FileCategory.README.value == "readme"
        assert FileCategory.TUTORIAL.value == "tutorial"
        assert FileCategory.USERGUIDE.value == "userguide"
        assert FileCategory.INSTALLATION.value == "installation"

    def test_all_values(self):
        values = FileCategory.all_values()
        assert len(values) == 4
        assert "readme" in values


class TestErrorCategory:
    def test_basic_categories(self):
        basic = ErrorCategory.basic_categories()
        assert "typo" in basic
        assert "link" in basic
        assert "duplicate" in basic

    def test_biology_categories(self):
        bio = ErrorCategory.biology_categories()
        assert "bio_term" in bio
        assert "function" in bio
        assert "gene_symbol_case" in bio

    def test_cli_config_categories(self):
        cli = ErrorCategory.cli_config_categories()
        assert "param_name" in cli
        assert "default_value" in cli

    def test_no_overlap_between_category_groups(self):
        basic = ErrorCategory.basic_categories()
        bio = ErrorCategory.biology_categories()
        cli = ErrorCategory.cli_config_categories()

        assert len(basic & bio) == 0
        assert len(basic & cli) == 0
        assert len(bio & cli) == 0


class TestFixStatus:
    def test_all_statuses(self):
        assert FixStatus.FIXED_TO_BASELINE.value == "fixed_to_baseline"
        assert FixStatus.FIXED_TO_VALID.value == "fixed_to_valid"
        assert FixStatus.UNCHANGED.value == "unchanged"
        assert FixStatus.WORSENED.value == "worsened"


class TestLLMGenerationConfig:
    def test_default_values(self):
        config = LLMGenerationConfig()
        assert config.model_name == "gpt-4o"
        assert config.max_tokens == GenerationLimits.MAX_TOKENS
        assert config.debug_enabled is False

    def test_from_environment_defaults(self):
        config = LLMGenerationConfig.from_environment()
        assert config.model_name == "gpt-4o"
        assert config.debug_enabled is False

    def test_content_limits(self):
        config = LLMGenerationConfig()
        assert (
            config.content_truncation_threshold == GenerationLimits.CONTENT_TRUNCATION
        )
        assert config.context_window_size == GenerationLimits.CONTEXT_WINDOW
        assert config.chunk_size == GenerationLimits.CHUNK_SIZE

    def test_custom_values(self):
        config = LLMGenerationConfig(
            model_name="gpt-4-turbo",
            max_tokens=8000,
            temperature=0.5,
            debug_enabled=True,
        )
        assert config.model_name == "gpt-4-turbo"
        assert config.max_tokens == 8000
        assert config.temperature == 0.5
        assert config.debug_enabled is True


class TestBenchmarkConfig:
    def test_default_stress_levels(self):
        config = BenchmarkConfig()
        assert config.stress_levels == [10, 20, 40, 60, 100]

    def test_default_max_workers(self):
        config = BenchmarkConfig()
        assert config.max_workers == 4

    def test_default_max_files_per_category(self):
        config = BenchmarkConfig()
        assert config.max_files_per_category == 10

    def test_custom_stress_levels(self):
        config = BenchmarkConfig(stress_levels=[5, 15, 25])
        assert config.stress_levels == [5, 15, 25]


class TestInjectionConfig:
    def test_defaults(self):
        config = InjectionConfig()
        assert config.min_per_category == 3
        assert config.max_words == 450
        assert config.min_similarity == 0.75

    def test_custom_values(self):
        config = InjectionConfig(
            min_per_category=5,
            max_words=600,
            preserve_keywords=["important", "keep"],
        )
        assert config.min_per_category == 5
        assert config.max_words == 600
        assert config.preserve_keywords == ["important", "keep"]


class TestMetricsConfig:
    def test_defaults(self):
        config = MetricsConfig()
        assert config.compute_f_score is True
        assert config.detect_semantic_fp is False

    def test_custom_values(self):
        config = MetricsConfig(
            compute_f_score=False,
            detect_semantic_fp=True,
            similarity_threshold=0.9,
        )
        assert config.compute_f_score is False
        assert config.detect_semantic_fp is True
        assert config.similarity_threshold == 0.9


class TestGenerationLimits:
    def test_limits_are_reasonable(self):
        assert GenerationLimits.MAX_TOKENS > 0
        assert GenerationLimits.CONTENT_TRUNCATION > 0
        assert GenerationLimits.CONTEXT_WINDOW > 0
        assert GenerationLimits.CHUNK_SIZE > 0
        assert GenerationLimits.MIN_SECTION_LENGTH > 0
        assert GenerationLimits.MAX_CONTINUATION_ATTEMPTS > 0

    def test_limits_hierarchy(self):
        # Chunk size should be less than context window
        assert GenerationLimits.CHUNK_SIZE <= GenerationLimits.CONTEXT_WINDOW
