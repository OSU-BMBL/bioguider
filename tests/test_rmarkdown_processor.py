"""Unit tests for RMarkdown processor."""

import pytest
from bioguider.generation.rmarkdown_processor import (
    RMarkdownProcessor,
    RMarkdownChunk,
    ChunkType,
    ChunkingResult,
)


SAMPLE_RMD = """---
title: "Test Document"
output: html_document
---

# Introduction

This is some introductory text.

```{r setup, include=FALSE}
knitr::opts_chunk$set(echo = TRUE)
```

## Analysis

More text here explaining the analysis.

```{r analysis}
library(dplyr)
data %>% filter(x > 5)
```

Final paragraph.
"""


SAMPLE_RMD_NO_YAML = """# Introduction

Some text.

```{r chunk1}
x <- 1
```

More text.
"""


SAMPLE_RMD_NESTED = """---
title: "Test"
---

Text before.

```{r}
# This is R code
print("hello")
```

Text between.

```python
# This is Python
print("world")
```

Text after.
"""


class TestRMarkdownProcessor:
    def test_split_basic_document(self):
        processor = RMarkdownProcessor()
        result = processor.split_into_chunks(SAMPLE_RMD)

        assert isinstance(result, ChunkingResult)
        assert len(result.chunks) > 0
        assert result.has_yaml_frontmatter is True
        assert len(result.warnings) == 0

    def test_split_identifies_yaml(self):
        processor = RMarkdownProcessor()
        result = processor.split_into_chunks(SAMPLE_RMD)

        yaml_chunks = [c for c in result.chunks if c.chunk_type == ChunkType.YAML]
        assert len(yaml_chunks) == 1
        assert 'title: "Test Document"' in yaml_chunks[0].content

    def test_split_identifies_code_blocks(self):
        processor = RMarkdownProcessor()
        result = processor.split_into_chunks(SAMPLE_RMD)

        code_chunks = [c for c in result.chunks if c.chunk_type == ChunkType.CODE]
        assert len(code_chunks) == 2

        # Check first code block
        assert "```{r setup" in code_chunks[0].content
        assert "knitr::opts_chunk" in code_chunks[0].content

        # Check second code block
        assert "```{r analysis}" in code_chunks[1].content
        assert "library(dplyr)" in code_chunks[1].content

    def test_split_identifies_text(self):
        processor = RMarkdownProcessor()
        result = processor.split_into_chunks(SAMPLE_RMD)

        text_chunks = [c for c in result.chunks if c.chunk_type == ChunkType.TEXT]
        assert len(text_chunks) >= 2

    def test_split_no_yaml(self):
        processor = RMarkdownProcessor()
        result = processor.split_into_chunks(SAMPLE_RMD_NO_YAML)

        assert result.has_yaml_frontmatter is False
        yaml_chunks = [c for c in result.chunks if c.chunk_type == ChunkType.YAML]
        assert len(yaml_chunks) == 0

    def test_split_empty_content(self):
        processor = RMarkdownProcessor()
        result = processor.split_into_chunks("")

        assert len(result.chunks) == 0
        assert result.total_lines == 0

    def test_code_fence_count(self):
        processor = RMarkdownProcessor()
        result = processor.split_into_chunks(SAMPLE_RMD)

        # SAMPLE_RMD has 2 code blocks = 4 fences
        assert result.code_fence_count == 4

    def test_validate_code_blocks_valid(self):
        processor = RMarkdownProcessor()
        original = "```{r}\nx <- 1\n```"
        generated = "```{r}\nx <- 2\n```"

        is_valid, error = processor.validate_code_blocks(original, generated)
        assert is_valid is True
        assert error == ""

    def test_validate_code_blocks_missing_fence(self):
        processor = RMarkdownProcessor()
        original = "```{r}\nx <- 1\n```"
        generated = "```{r}\nx <- 2"  # Missing closing fence

        is_valid, error = processor.validate_code_blocks(original, generated)
        assert is_valid is False
        assert "mismatch" in error.lower()

    def test_validate_code_blocks_extra_fence(self):
        processor = RMarkdownProcessor()
        original = "```{r}\nx <- 1\n```"
        generated = "```{r}\nx <- 1\n```\n```python\nprint(1)\n```"  # Extra block

        is_valid, error = processor.validate_code_blocks(original, generated)
        assert is_valid is False

    def test_merge_chunks(self):
        processor = RMarkdownProcessor()
        chunks = [
            RMarkdownChunk(ChunkType.YAML, "---\ntitle: Test\n---"),
            RMarkdownChunk(ChunkType.TEXT, "# Intro\nText"),
            RMarkdownChunk(ChunkType.CODE, "```{r}\nx <- 1\n```"),
        ]

        merged = processor.merge_chunks(chunks)
        assert "---\ntitle: Test\n---" in merged
        assert "# Intro" in merged
        assert "```{r}" in merged

    def test_legacy_format(self):
        processor = RMarkdownProcessor()
        chunks = processor.split_into_chunks_legacy(SAMPLE_RMD)

        assert isinstance(chunks, list)
        assert all(isinstance(c, dict) for c in chunks)
        assert all("type" in c and "content" in c for c in chunks)

    def test_get_chunk_summary(self):
        processor = RMarkdownProcessor()
        summary = processor.get_chunk_summary(SAMPLE_RMD)

        assert summary["total_chunks"] > 0
        assert summary["yaml_chunks"] == 1
        assert summary["code_chunks"] == 2
        assert summary["has_yaml_frontmatter"] is True

    def test_contains_code_fences(self):
        processor = RMarkdownProcessor()
        assert processor.contains_code_fences("```{r}\nx\n```") is True
        assert processor.contains_code_fences("No fences here") is False

    def test_is_rmarkdown_file(self):
        processor = RMarkdownProcessor()
        assert processor.is_rmarkdown_file("test.Rmd") is True
        assert processor.is_rmarkdown_file("test.rmd") is True
        assert processor.is_rmarkdown_file("test.qmd") is True
        assert processor.is_rmarkdown_file("test.md") is False
        assert processor.is_rmarkdown_file("test.py") is False

    def test_split_preserves_content(self):
        """Ensure splitting and merging preserves original content."""
        processor = RMarkdownProcessor()
        result = processor.split_into_chunks(SAMPLE_RMD)
        merged = processor.merge_chunks(result.chunks)

        # Content should be identical when merged back
        assert merged.strip() == SAMPLE_RMD.strip()

    def test_multiple_code_languages(self):
        processor = RMarkdownProcessor()
        result = processor.split_into_chunks(SAMPLE_RMD_NESTED)

        code_chunks = [c for c in result.chunks if c.chunk_type == ChunkType.CODE]
        assert len(code_chunks) == 2

        # One R, one Python
        assert any("```{r}" in c.content for c in code_chunks)
        assert any("```python" in c.content for c in code_chunks)

    def test_unclosed_code_block_warning(self):
        processor = RMarkdownProcessor()
        unclosed = "# Title\n\n```{r}\nx <- 1\n# No closing fence"

        result = processor.split_into_chunks(unclosed)
        assert len(result.warnings) > 0
        assert any("unclosed" in w.lower() for w in result.warnings)
