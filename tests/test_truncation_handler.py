"""
Tests for TruncationHandler module.
"""

import pytest
from bioguider.generation.truncation_handler import (
    TruncationHandler,
    TruncationResult,
    CompletenessResult,
    FileType,
)


class TestFileType:
    """Tests for FileType enum."""

    def test_file_type_values(self):
        """Test FileType enum has expected values."""
        assert FileType.RMARKDOWN == "rmarkdown"
        assert FileType.MARKDOWN == "markdown"
        assert FileType.PYTHON == "python"
        assert FileType.JAVASCRIPT == "javascript"
        assert FileType.OTHER == "other"


class TestTruncationResult:
    """Tests for TruncationResult dataclass."""

    def test_truncation_result_defaults(self):
        """Test TruncationResult default values."""
        result = TruncationResult(is_truncated=False)
        assert result.is_truncated is False
        assert result.reason == ""
        assert result.detected_issue == ""

    def test_truncation_result_with_values(self):
        """Test TruncationResult with all values."""
        result = TruncationResult(
            is_truncated=True,
            reason="Content too short",
            detected_issue="Length 50 < 100",
        )
        assert result.is_truncated is True
        assert result.reason == "Content too short"
        assert result.detected_issue == "Length 50 < 100"


class TestCompletenessResult:
    """Tests for CompletenessResult dataclass."""

    def test_completeness_result_defaults(self):
        """Test CompletenessResult default values."""
        result = CompletenessResult(is_complete=True)
        assert result.is_complete is True
        assert result.reason == ""
        assert result.confidence == 0.0

    def test_completeness_result_with_values(self):
        """Test CompletenessResult with all values."""
        result = CompletenessResult(
            is_complete=True, reason="Has conclusion", confidence=0.9
        )
        assert result.is_complete is True
        assert result.reason == "Has conclusion"
        assert result.confidence == 0.9


class TestTruncationHandlerDetectFileType:
    """Tests for TruncationHandler._detect_file_type."""

    def test_detect_rmarkdown(self):
        """Test detection of RMarkdown files."""
        handler = TruncationHandler()
        assert handler._detect_file_type("test.Rmd") == FileType.RMARKDOWN
        assert handler._detect_file_type("test.rmd") == FileType.RMARKDOWN
        assert handler._detect_file_type("test.qmd") == FileType.RMARKDOWN

    def test_detect_markdown(self):
        """Test detection of Markdown files."""
        handler = TruncationHandler()
        assert handler._detect_file_type("README.md") == FileType.MARKDOWN
        assert handler._detect_file_type("docs.rst") == FileType.MARKDOWN
        assert handler._detect_file_type("notes.txt") == FileType.MARKDOWN

    def test_detect_python(self):
        """Test detection of Python files."""
        handler = TruncationHandler()
        assert handler._detect_file_type("script.py") == FileType.PYTHON

    def test_detect_javascript(self):
        """Test detection of JavaScript files."""
        handler = TruncationHandler()
        assert handler._detect_file_type("app.js") == FileType.JAVASCRIPT
        assert handler._detect_file_type("app.ts") == FileType.JAVASCRIPT
        assert handler._detect_file_type("component.jsx") == FileType.JAVASCRIPT
        assert handler._detect_file_type("component.tsx") == FileType.JAVASCRIPT

    def test_detect_other(self):
        """Test detection of other file types."""
        handler = TruncationHandler()
        assert handler._detect_file_type("data.json") == FileType.OTHER
        assert handler._detect_file_type("style.css") == FileType.OTHER
        assert handler._detect_file_type("unknown") == FileType.OTHER


class TestTruncationHandlerDetectTruncation:
    """Tests for TruncationHandler.detect_truncation."""

    def test_empty_content_is_truncated(self):
        """Test that empty content is detected as truncated."""
        handler = TruncationHandler()
        result = handler.detect_truncation("", "test.md")
        assert result.is_truncated is True
        assert "too short" in result.reason.lower()

    def test_very_short_content_is_truncated(self):
        """Test that very short content is detected as truncated."""
        handler = TruncationHandler()
        result = handler.detect_truncation("Hello world", "test.md")
        assert result.is_truncated is True

    def test_short_content_below_threshold(self):
        """Test content below short threshold is truncated."""
        handler = TruncationHandler()
        content = (
            "a" * 200
        )  # Above MIN_CONTENT_LENGTH but below SHORT_CONTENT_THRESHOLD
        result = handler.detect_truncation(content, "test.md")
        assert result.is_truncated is True
        assert "short threshold" in result.reason.lower()

    def test_unbalanced_code_fences(self):
        """Test that unbalanced code fences are detected."""
        handler = TruncationHandler()
        content = "a" * 600 + "\n```python\ncode here"  # One fence, no closing
        result = handler.detect_truncation(content, "test.md")
        assert result.is_truncated is True
        assert "code fence" in result.reason.lower()

    def test_balanced_code_fences_ok(self):
        """Test that balanced code fences are not flagged."""
        handler = TruncationHandler()
        content = "a" * 600 + "\n```python\ncode here\n```"
        result = handler.detect_truncation(content, "test.md")
        assert result.is_truncated is False

    def test_original_content_comparison_truncated(self):
        """Test truncation detection based on original content length."""
        handler = TruncationHandler()
        original = "x" * 10000
        generated = "y" * 600  # Much shorter than 80% of original
        result = handler.detect_truncation(generated, "test.md", original)
        assert result.is_truncated is True
        assert "shorter than original" in result.reason.lower()

    def test_original_content_comparison_ok(self):
        """Test non-truncation when generated is close to original length."""
        handler = TruncationHandler()
        original = "x" * 1000
        generated = "y" * 900  # 90% of original, above 80% threshold
        result = handler.detect_truncation(generated, "test.md", original)
        assert result.is_truncated is False

    def test_reasonable_content_not_truncated(self):
        """Test that reasonable content is not detected as truncated."""
        handler = TruncationHandler()
        content = "This is a paragraph.\n\n" * 50  # Well-formed content
        result = handler.detect_truncation(content, "test.md")
        assert result.is_truncated is False


class TestTruncationHandlerRMarkdown:
    """Tests for RMarkdown-specific truncation detection."""

    def test_unclosed_r_chunk(self):
        """Test detection of unclosed R chunk."""
        handler = TruncationHandler()
        content = "a" * 600 + "\n```{r setup}\nlibrary(tidyverse)\n"
        result = handler.detect_truncation(content, "test.Rmd")
        assert result.is_truncated is True
        # May detect as unbalanced code fences or unclosed R chunk
        assert "chunk" in result.reason.lower() or "fence" in result.reason.lower()

    def test_closed_r_chunk_ok(self):
        """Test that closed R chunk is not flagged."""
        handler = TruncationHandler()
        content = "a" * 600 + "\n```{r setup}\nlibrary(tidyverse)\n```"
        result = handler.detect_truncation(content, "test.Rmd")
        assert result.is_truncated is False


class TestTruncationHandlerPython:
    """Tests for Python-specific truncation detection."""

    def test_incomplete_function_definition(self):
        """Test detection of incomplete function definition."""
        handler = TruncationHandler()
        content = "a" * 600 + "\n\ndef my_function():"
        result = handler.detect_truncation(content, "script.py")
        assert result.is_truncated is True
        assert "Python" in result.reason or "incomplete" in result.reason.lower()

    def test_incomplete_class_definition(self):
        """Test detection of incomplete class definition."""
        handler = TruncationHandler()
        content = "a" * 600 + "\n\nclass MyClass:"
        result = handler.detect_truncation(content, "script.py")
        assert result.is_truncated is True


class TestTruncationHandlerMarkdown:
    """Tests for Markdown-specific truncation detection."""

    def test_incomplete_header(self):
        """Test detection of incomplete header."""
        handler = TruncationHandler()
        content = "a" * 600 + "\n\n## "
        result = handler.detect_truncation(content, "README.md")
        assert result.is_truncated is True
        assert "markdown" in result.reason.lower()

    def test_ends_with_list_marker(self):
        """Test detection of content ending with list marker."""
        handler = TruncationHandler()
        content = "a" * 600 + "\n\n-"
        result = handler.detect_truncation(content, "README.md")
        assert result.is_truncated is True


class TestTruncationHandlerContinuationPoint:
    """Tests for TruncationHandler.find_continuation_point."""

    def test_empty_content_returns_none(self):
        """Test that empty content returns None."""
        handler = TruncationHandler()
        result = handler.find_continuation_point("")
        assert result is None

    def test_short_content_returns_none(self):
        """Test that very short content returns None."""
        handler = TruncationHandler()
        result = handler.find_continuation_point("line1\nline2")
        assert result is None

    def test_find_last_section(self):
        """Test finding last complete section."""
        handler = TruncationHandler()
        content = (
            "\n".join([f"line {i}" for i in range(20)]) + "\n## Section\nContent here"
        )
        result = handler.find_continuation_point(content)
        assert result is not None
        assert "## Section" in result

    def test_find_last_paragraph(self):
        """Test finding last complete paragraph."""
        handler = TruncationHandler()
        lines = [f"line {i}" for i in range(15)]
        lines.append("This is a complete sentence.")
        lines.append("Incomplete")
        content = "\n".join(lines)
        result = handler.find_continuation_point(content)
        assert result is not None


class TestTruncationHandlerCompleteness:
    """Tests for TruncationHandler.check_completeness."""

    def test_empty_content_incomplete(self):
        """Test that empty content is incomplete."""
        handler = TruncationHandler()
        result = handler.check_completeness("", "test.md")
        assert result.is_complete is False
        assert result.confidence == 0.0

    def test_short_content_incomplete(self):
        """Test that short content is incomplete."""
        handler = TruncationHandler()
        result = handler.check_completeness("Hello", "test.md")
        assert result.is_complete is False

    def test_length_ratio_check(self):
        """Test completeness based on length ratio."""
        handler = TruncationHandler()
        original = "x" * 10000
        generated = "y" * 5000  # Only 50%, below 90% threshold
        result = handler.check_completeness(generated, "test.md", original)
        assert result.is_complete is False
        assert "ratio" in result.reason.lower()

    def test_unbalanced_code_blocks_incomplete(self):
        """Test that unbalanced code blocks make content incomplete."""
        handler = TruncationHandler()
        content = "a" * 500 + "\n```python\ncode"  # Unbalanced
        result = handler.check_completeness(content, "test.md")
        assert result.is_complete is False
        assert "code block" in result.reason.lower()

    def test_rmarkdown_with_conclusion_complete(self):
        """Test RMarkdown with conclusion markers is complete."""
        handler = TruncationHandler()
        content = (
            "---\ntitle: Test\n---\n"
            + "a" * 3000
            + "\n\n## Conclusion\n\nThis is done.\n```{r}\n1+1\n```"
        )
        result = handler.check_completeness(content, "vignette.Rmd")
        assert result.is_complete is True
        assert result.confidence >= 0.8

    def test_rmarkdown_without_frontmatter_incomplete(self):
        """Test RMarkdown without YAML frontmatter - caught by RMarkdown check."""
        handler = TruncationHandler()
        # Use content that won't pass generic completeness check either
        content = "## Header\n" + "a" * 3000 + "\n## Conclusion\nIncomplete"
        result = handler.check_completeness(content, "test.Rmd")
        # RMarkdown check catches missing frontmatter before generic check runs
        assert result.is_complete is False

    def test_markdown_with_conclusion_complete(self):
        """Test Markdown with conclusion section is complete."""
        handler = TruncationHandler()
        content = (
            "# Title\n" + "a" * 3000 + "\n\n## Conclusion\n\nThis document is complete."
        )
        result = handler.check_completeness(content, "README.md")
        assert result.is_complete is True

    def test_python_balanced_complete(self):
        """Test Python with balanced structure is complete."""
        handler = TruncationHandler()
        content = "\n".join(
            [
                "def hello():",
                "    print('Hello')",
                "",
                "def world():",
                "    return 'World'",
            ]
            + ["# comment"] * 20
            + ["result = hello()"]
        )
        result = handler.check_completeness(content, "script.py")
        assert result.is_complete is True

    def test_python_unbalanced_incomplete(self):
        """Test Python with unbalanced brackets is incomplete."""
        handler = TruncationHandler()
        content = "\n".join(
            [
                "def hello():",
                "    data = [1, 2, 3",  # Missing closing bracket
            ]
            + ["# padding"] * 25
        )
        result = handler.check_completeness(content, "script.py")
        assert result.is_complete is False
        assert "bracket" in result.reason.lower()

    def test_javascript_balanced_complete(self):
        """Test JavaScript with balanced structure is complete."""
        handler = TruncationHandler()
        content = "\n".join(
            [
                "function hello() {",
                "    console.log('Hello');",
                "}",
            ]
            + ["// comment"] * 20
            + ["hello();"]
        )
        result = handler.check_completeness(content, "app.js")
        assert result.is_complete is True

    def test_generic_ends_with_period_complete(self):
        """Test generic content ending with period."""
        handler = TruncationHandler()
        content = "a" * 4000 + "\n\nThis is the end."
        result = handler.check_completeness(content, "unknown.xyz")
        assert result.is_complete is True


class TestTruncationHandlerConvenienceMethods:
    """Tests for convenience methods is_truncated and is_complete."""

    def test_is_truncated_method(self):
        """Test is_truncated convenience method."""
        handler = TruncationHandler()
        assert handler.is_truncated("short", "test.md") is True
        assert handler.is_truncated("a" * 600, "test.md") is False

    def test_is_complete_method(self):
        """Test is_complete convenience method."""
        handler = TruncationHandler()
        assert handler.is_complete("short", "test.md") is False
        content = (
            "---\ntitle: T\n---\n"
            + "a" * 3500
            + "\n## Conclusion\nDone.\n```{r}\n1\n```"
        )
        assert handler.is_complete(content, "test.Rmd") is True


class TestTruncationHandlerEdgeCases:
    """Tests for edge cases."""

    def test_none_content_handling(self):
        """Test handling of None-like content."""
        handler = TruncationHandler()
        # Empty string
        result = handler.detect_truncation("", "test.md")
        assert result.is_truncated is True

    def test_whitespace_only_content(self):
        """Test handling of whitespace-only content."""
        handler = TruncationHandler()
        result = handler.detect_truncation("   \n\n   ", "test.md")
        assert result.is_truncated is True

    def test_content_with_special_characters(self):
        """Test handling of content with special characters."""
        handler = TruncationHandler()
        content = "Unicode test: \u4e2d\u6587 \u65e5\u672c\u8a9e\n" * 100
        result = handler.detect_truncation(content, "test.md")
        # Should handle gracefully
        assert isinstance(result, TruncationResult)

    def test_very_long_content(self):
        """Test handling of very long content."""
        handler = TruncationHandler()
        content = "a" * 100000 + "\n\nThe end."
        result = handler.check_completeness(content, "test.md")
        assert result.is_complete is True
