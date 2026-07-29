"""
RMarkdown processing utilities for document generation.

This module handles RMarkdown-specific operations:
- Splitting documents into chunks (YAML, code, text)
- Validating code block structure
- Merging processed chunks back together
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple


class ChunkType(str, Enum):
    """Types of chunks in an RMarkdown document."""

    YAML = "yaml"
    CODE = "code"
    TEXT = "text"


@dataclass
class RMarkdownChunk:
    """A single chunk from an RMarkdown document."""

    chunk_type: ChunkType
    content: str
    line_start: int = 0
    line_end: int = 0

    def to_dict(self) -> dict:
        """Convert to dictionary for backward compatibility."""
        return {
            "type": self.chunk_type.value,
            "content": self.content,
        }


@dataclass
class ChunkingResult:
    """Result of splitting an RMarkdown document."""

    chunks: List[RMarkdownChunk]
    total_lines: int
    code_fence_count: int
    has_yaml_frontmatter: bool
    warnings: List[str]


class RMarkdownProcessor:
    """
    Processes RMarkdown documents for chunk-based generation.

    Key responsibilities:
    - Split documents into processable chunks
    - Preserve code blocks exactly
    - Validate code fence structure
    - Merge chunks back together
    """

    def split_into_chunks(self, content: str) -> ChunkingResult:
        """
        Split RMarkdown content into chunks for processing.

        CRITICAL: This function must correctly identify code blocks to preserve them.
        Code blocks in RMarkdown start with ```{r...} or ``` and end with ```.

        Args:
            content: RMarkdown document content

        Returns:
            ChunkingResult with list of chunks and metadata
        """
        chunks: List[RMarkdownChunk] = []
        warnings: List[str] = []

        if not content:
            return ChunkingResult(
                chunks=[],
                total_lines=0,
                code_fence_count=0,
                has_yaml_frontmatter=False,
                warnings=[],
            )

        lines = content.split("\n")
        n = len(lines)
        i = 0
        has_yaml = False

        # Handle YAML frontmatter
        if n >= 3 and lines[0].strip() == "---":
            j = 1
            while j < n and lines[j].strip() != "---":
                j += 1
            if j < n and lines[j].strip() == "---":
                yaml_content = "\n".join(lines[0 : j + 1])
                chunks.append(
                    RMarkdownChunk(
                        chunk_type=ChunkType.YAML,
                        content=yaml_content,
                        line_start=0,
                        line_end=j,
                    )
                )
                i = j + 1
                has_yaml = True

        buffer: List[str] = []
        buffer_start = i
        in_code = False

        for k in range(i, n):
            line = lines[k]
            stripped = line.strip()

            # Detect code fence: ``` or ```{r...} or ```python etc
            is_code_fence = stripped.startswith("```")

            if is_code_fence:
                if in_code:
                    # This is a closing fence
                    buffer.append(line)
                    chunks.append(
                        RMarkdownChunk(
                            chunk_type=ChunkType.CODE,
                            content="\n".join(buffer),
                            line_start=buffer_start,
                            line_end=k,
                        )
                    )
                    buffer = []
                    buffer_start = k + 1
                    in_code = False
                else:
                    # This is an opening fence
                    # Save any accumulated text first
                    if buffer and any(s.strip() for s in buffer):
                        chunks.append(
                            RMarkdownChunk(
                                chunk_type=ChunkType.TEXT,
                                content="\n".join(buffer),
                                line_start=buffer_start,
                                line_end=k - 1,
                            )
                        )
                    # Start new code block with the opening fence
                    buffer = [line]
                    buffer_start = k
                    in_code = True
            else:
                buffer.append(line)

        # Handle remaining buffer
        if buffer and any(s.strip() for s in buffer):
            if in_code:
                warnings.append("Unclosed code block detected in RMarkdown")
                chunks.append(
                    RMarkdownChunk(
                        chunk_type=ChunkType.CODE,
                        content="\n".join(buffer),
                        line_start=buffer_start,
                        line_end=n - 1,
                    )
                )
            else:
                chunks.append(
                    RMarkdownChunk(
                        chunk_type=ChunkType.TEXT,
                        content="\n".join(buffer),
                        line_start=buffer_start,
                        line_end=n - 1,
                    )
                )

        # Count code fences for validation
        original_fences = len(re.findall(r"^```", content, flags=re.M))
        chunk_fences = 0
        for ch in chunks:
            if ch.chunk_type == ChunkType.CODE:
                chunk_fences += len(re.findall(r"^```", ch.content, flags=re.M))

        if original_fences != chunk_fences:
            warnings.append(
                f"Code fence count mismatch: original={original_fences}, chunks={chunk_fences}"
            )

        return ChunkingResult(
            chunks=chunks,
            total_lines=n,
            code_fence_count=original_fences,
            has_yaml_frontmatter=has_yaml,
            warnings=warnings,
        )

    def split_into_chunks_legacy(self, content: str) -> List[dict]:
        """
        Split content and return in legacy format for backward compatibility.

        Returns list of dicts with 'type' and 'content' keys.
        """
        result = self.split_into_chunks(content)
        return [chunk.to_dict() for chunk in result.chunks]

    def validate_code_blocks(self, original: str, generated: str) -> Tuple[bool, str]:
        """
        Validate that code block structure is preserved.

        Args:
            original: Original document content
            generated: Generated/modified content

        Returns:
            Tuple of (is_valid, error_message)
        """
        original_fences = len(re.findall(r"^```", original, flags=re.M))
        generated_fences = len(re.findall(r"^```", generated, flags=re.M))

        if original_fences != generated_fences:
            return False, (
                f"Code block count mismatch: "
                f"original={original_fences}, generated={generated_fences}"
            )

        # Check RMarkdown chunk headers specifically
        original_rmd = re.findall(r"^```\{[^}]*\}", original, flags=re.M)
        generated_rmd = re.findall(r"^```\{[^}]*\}", generated, flags=re.M)

        if len(original_rmd) != len(generated_rmd):
            return False, (
                f"RMarkdown chunk header count mismatch: "
                f"original={len(original_rmd)}, generated={len(generated_rmd)}"
            )

        # Check closing fences match opening fences
        original_close = len(re.findall(r"^```\s*$", original, flags=re.M))
        generated_close = len(re.findall(r"^```\s*$", generated, flags=re.M))

        if original_close != generated_close:
            return False, (
                f"Closing fence count mismatch: "
                f"original={original_close}, generated={generated_close}"
            )

        return True, ""

    def merge_chunks(self, chunks: List[RMarkdownChunk]) -> str:
        """
        Merge chunks back into a single document.

        Args:
            chunks: List of processed chunks

        Returns:
            Merged document content
        """
        return "\n".join(chunk.content for chunk in chunks)

    def merge_chunks_legacy(self, chunks: List[dict]) -> str:
        """
        Merge chunks from legacy format (list of dicts).

        Args:
            chunks: List of dicts with 'content' key

        Returns:
            Merged document content
        """
        return "\n".join(chunk.get("content", "") for chunk in chunks)

    def get_chunk_summary(self, content: str) -> dict:
        """
        Get a summary of chunks in the document.

        Args:
            content: RMarkdown document content

        Returns:
            Summary dict with counts by type
        """
        result = self.split_into_chunks(content)

        summary = {
            "total_chunks": len(result.chunks),
            "yaml_chunks": 0,
            "code_chunks": 0,
            "text_chunks": 0,
            "code_fence_count": result.code_fence_count,
            "has_yaml_frontmatter": result.has_yaml_frontmatter,
            "warnings": result.warnings,
        }

        for chunk in result.chunks:
            if chunk.chunk_type == ChunkType.YAML:
                summary["yaml_chunks"] += 1
            elif chunk.chunk_type == ChunkType.CODE:
                summary["code_chunks"] += 1
            elif chunk.chunk_type == ChunkType.TEXT:
                summary["text_chunks"] += 1

        return summary

    def contains_code_fences(self, text: str) -> bool:
        """Check if text contains code fences."""
        return "```" in text

    def is_rmarkdown_file(self, filepath: str) -> bool:
        """Check if a file is an RMarkdown file based on extension."""
        return filepath.lower().endswith((".rmd", ".qmd"))
