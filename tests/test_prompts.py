"""Unit tests for prompt template management."""

import pytest
from bioguider.generation.prompts import (
    PromptLoader,
    PromptTemplate,
    load_prompt,
    format_prompt,
)


class TestPromptLoader:
    def test_load_section_template(self):
        loader = PromptLoader()
        template = loader.load(PromptTemplate.SECTION)
        assert isinstance(template, str)
        assert len(template) > 100
        assert "{suggestion_category}" in template
        assert "{guidance}" in template

    def test_load_full_document_template(self):
        loader = PromptLoader()
        template = loader.load(PromptTemplate.FULL_DOCUMENT)
        assert isinstance(template, str)
        assert "{evaluation_report}" in template
        assert "{target_file}" in template

    def test_load_readme_comprehensive_template(self):
        loader = PromptLoader()
        template = loader.load(PromptTemplate.README_COMPREHENSIVE)
        assert isinstance(template, str)
        assert "README" in template

    def test_load_continuation_template(self):
        loader = PromptLoader()
        template = loader.load(PromptTemplate.CONTINUATION)
        assert isinstance(template, str)
        assert "{existing_content_tail}" in template

    def test_format_section_template(self):
        loader = PromptLoader()
        formatted = loader.format(
            PromptTemplate.SECTION,
            suggestion_category="installation",
            anchor_title="Installation",
            guidance="Add installation instructions",
            original_text="",
            evaluation_score="Fair",
            context="Sample context",
            tone_markers="professional",
            heading_style="#",
            list_style="-",
            link_style="inline",
        )
        assert "installation" in formatted
        assert "Installation" in formatted
        assert "Add installation instructions" in formatted

    def test_caching(self):
        loader = PromptLoader()
        # First load
        template1 = loader.load(PromptTemplate.SECTION)
        # Second load should use cache
        template2 = loader.load(PromptTemplate.SECTION)
        assert template1 is template2  # Same object from cache

    def test_clear_cache(self):
        loader = PromptLoader()
        loader.load(PromptTemplate.SECTION)
        assert PromptTemplate.SECTION in loader._cache
        loader.clear_cache()
        assert len(loader._cache) == 0

    def test_get_available_templates(self):
        loader = PromptLoader()
        templates = loader.get_available_templates()
        assert PromptTemplate.SECTION in templates
        assert PromptTemplate.FULL_DOCUMENT in templates
        assert PromptTemplate.CONTINUATION in templates

    def test_load_nonexistent_template(self):
        loader = PromptLoader()
        with pytest.raises(FileNotFoundError):
            loader.load("nonexistent_template")


class TestConvenienceFunctions:
    def test_load_prompt(self):
        template = load_prompt(PromptTemplate.SECTION)
        assert isinstance(template, str)
        assert len(template) > 100

    def test_format_prompt(self):
        formatted = format_prompt(
            PromptTemplate.CONTINUATION,
            existing_content_tail="Last part of content...",
        )
        assert "Last part of content..." in formatted
