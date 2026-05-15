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


class TestFullDocumentConsistencyCarveOut:
    """Pin the prompt language that lets the generator actually act on
    Consistency suggestions. Earlier benchmark runs showed the LLM ignoring
    the suggestions because the no-delete rules were too strict; the
    Consistency bullet needs explicit REQUIRED authority and a concrete
    before/after example to override them."""

    def _template(self) -> str:
        return PromptLoader().load(PromptTemplate.FULL_DOCUMENT)

    def test_consistency_bullet_is_required(self):
        tpl = self._template()
        assert "Consistency (REQUIRED" in tpl, (
            "Consistency bullet must carry REQUIRED authority — without it the "
            "no-delete rules override it and the generator leaves bogus flags in place"
        )

    def test_consistency_has_concrete_before_after_example(self):
        tpl = self._template()
        # The before/after pair must demonstrate token-level removal so the
        # LLM has a worked example, not just an abstract instruction.
        assert "before:" in tpl and "after :" in tpl
        # The canonical example pins --cores 4 as the bogus flag.
        assert "--cores 4" in tpl, "the before example must show a bogus --cores flag"

    def test_no_delete_rule_is_narrowed_not_blanket(self):
        tpl = self._template()
        # The old blanket "Deleting ANY existing content" rule overrode the
        # Consistency bullet — make sure it has been narrowed.
        assert "Deleting ANY existing content" not in tpl
        assert "Deleting entire sections, paragraphs, or code blocks" in tpl

    def test_no_delete_rule_mentions_consistency_exception(self):
        tpl = self._template()
        # The narrowing must explicitly call out the Consistency carve-out
        # so the LLM doesn't read the narrow rule as still blocking token edits.
        assert "Consistency suggestions are explicitly permitted" in tpl

    def test_strict_constraints_require_consistency_application(self):
        tpl = self._template()
        # Positive imperative: Consistency suggestions must be applied,
        # not just allowed. This is what flips the LLM from "leave alone"
        # to "actually edit".
        assert "REQUIRED: Apply every Consistency suggestion" in tpl
