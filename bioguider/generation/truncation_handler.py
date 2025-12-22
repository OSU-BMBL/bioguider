"""
Truncation detection and handling for LLM content generation.

This module handles:
- Detecting when LLM output appears truncated
- Finding appropriate continuation points
- Checking if content appears complete
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Optional, List, Tuple


class FileType(str, Enum):
    """Detected file types for specialized handling."""

    RMARKDOWN = "rmarkdown"
    MARKDOWN = "markdown"
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    OTHER = "other"


@dataclass
class TruncationResult:
    """Result of truncation detection."""

    is_truncated: bool
    reason: str = ""
    detected_issue: str = ""


@dataclass
class CompletenessResult:
    """Result of completeness check."""

    is_complete: bool
    reason: str = ""
    confidence: float = 0.0


class TruncationHandler:
    """
    Handles truncation detection and continuation for LLM outputs.

    Key responsibilities:
    - Detect if content appears truncated
    - Find appropriate continuation points
    - Check if content appears complete
    """

    # Minimum content length thresholds
    MIN_CONTENT_LENGTH = 100
    SHORT_CONTENT_THRESHOLD = 500
    REASONABLE_LENGTH = 3000

    # Length ratio thresholds
    TRUNCATION_RATIO = 0.8  # If generated < 80% of original, likely truncated
    COMPLETION_RATIO = 0.9  # Need 90% of original to be considered complete

    def __init__(self):
        """Initialize the truncation handler."""
        pass

    def detect_truncation(
        self, content: str, target_file: str, original_content: Optional[str] = None
    ) -> TruncationResult:
        """
        Detect if content appears to be truncated based on common patterns.

        Args:
            content: Generated content to check
            target_file: Target file path for context
            original_content: Original content for comparison (if available)

        Returns:
            TruncationResult with detection details
        """
        if not content or len(content.strip()) < self.MIN_CONTENT_LENGTH:
            return TruncationResult(
                is_truncated=True,
                reason="Content too short",
                detected_issue=f"Length {len(content) if content else 0} < {self.MIN_CONTENT_LENGTH}",
            )

        # Compare to original length if available
        if original_content:
            original_len = len(original_content)
            generated_len = len(content)
            if generated_len < original_len * self.TRUNCATION_RATIO:
                return TruncationResult(
                    is_truncated=True,
                    reason="Significantly shorter than original",
                    detected_issue=f"Generated {generated_len} vs original {original_len}",
                )

        # Very short content
        if len(content) < self.SHORT_CONTENT_THRESHOLD:
            return TruncationResult(
                is_truncated=True,
                reason="Content below short threshold",
                detected_issue=f"Length {len(content)} < {self.SHORT_CONTENT_THRESHOLD}",
            )

        # Check for unbalanced code blocks
        code_fence_count = content.count("```")
        if code_fence_count > 0 and code_fence_count % 2 != 0:
            return TruncationResult(
                is_truncated=True,
                reason="Unbalanced code fences",
                detected_issue=f"Found {code_fence_count} code fences (odd number)",
            )

        # File-specific checks
        file_type = self._detect_file_type(target_file)

        if file_type == FileType.RMARKDOWN:
            result = self._check_rmarkdown_truncation(content)
            if result.is_truncated:
                return result

        if file_type == FileType.PYTHON:
            result = self._check_python_truncation(content)
            if result.is_truncated:
                return result

        if file_type in (FileType.MARKDOWN, FileType.RMARKDOWN):
            result = self._check_markdown_truncation(content)
            if result.is_truncated:
                return result

        return TruncationResult(is_truncated=False)

    def _detect_file_type(self, target_file: str) -> FileType:
        """Detect file type from extension."""
        lower = target_file.lower()
        if lower.endswith(".rmd") or lower.endswith(".qmd"):
            return FileType.RMARKDOWN
        elif lower.endswith(".md") or lower.endswith(".rst") or lower.endswith(".txt"):
            return FileType.MARKDOWN
        elif lower.endswith(".py"):
            return FileType.PYTHON
        elif lower.endswith((".js", ".ts", ".jsx", ".tsx")):
            return FileType.JAVASCRIPT
        return FileType.OTHER

    def _check_rmarkdown_truncation(self, content: str) -> TruncationResult:
        """Check RMarkdown-specific truncation patterns."""
        r_chunks_open = re.findall(r"```\{r[^}]*\}", content)
        if r_chunks_open and not content.rstrip().endswith("```"):
            return TruncationResult(
                is_truncated=True,
                reason="RMarkdown chunk not closed",
                detected_issue="Has R chunks but doesn't end with closing fence",
            )
        return TruncationResult(is_truncated=False)

    def _check_python_truncation(self, content: str) -> TruncationResult:
        """Check Python-specific truncation patterns."""
        lines = content.split("\n")
        last_lines = [line.strip() for line in lines[-5:] if line.strip()]
        if last_lines:
            last_line = last_lines[-1]
            if (
                last_line.endswith(":")
                or last_line.endswith("(")
                or "def " in last_line
                or "class " in last_line
            ):
                return TruncationResult(
                    is_truncated=True,
                    reason="Incomplete Python statement",
                    detected_issue=f"Last line: {last_line[:50]}",
                )
        return TruncationResult(is_truncated=False)

    def _check_markdown_truncation(self, content: str) -> TruncationResult:
        """Check Markdown-specific truncation patterns."""
        lines = content.split("\n")
        last_non_empty = None
        for line in reversed(lines):
            if line.strip():
                last_non_empty = line.strip()
                break

        if last_non_empty:
            incomplete_endings = ["##", "###", "####", "-", "*", ":", "|"]
            for ending in incomplete_endings:
                if last_non_empty.endswith(ending):
                    return TruncationResult(
                        is_truncated=True,
                        reason="Incomplete markdown structure",
                        detected_issue=f"Ends with '{ending}'",
                    )

            # Check end patterns
            content_end = content[-300:].strip().lower()
            incomplete_patterns = ["## ", "### ", "#### ", "```{", "```r", "```python"]
            for pattern in incomplete_patterns:
                if content_end.endswith(pattern.lower()):
                    return TruncationResult(
                        is_truncated=True,
                        reason="Incomplete markdown section",
                        detected_issue=f"Ends with '{pattern}'",
                    )

        return TruncationResult(is_truncated=False)

    def find_continuation_point(
        self, content: str, original_content: Optional[str] = None
    ) -> Optional[str]:
        """
        Find a suitable continuation point in the content.

        Args:
            content: The generated content so far
            original_content: The original content for comparison

        Returns:
            Content from continuation point, or None if not found
        """
        if not content:
            return None

        lines = content.split("\n")
        if len(lines) < 10:
            return None

        # Strategy 1: Find the last complete section
        continuation = self._find_last_complete_section(lines)
        if continuation:
            return continuation

        # Strategy 2: Find the last complete code block
        continuation = self._find_last_complete_code_block(lines)
        if continuation:
            return continuation

        # Strategy 3: Find last complete paragraph
        continuation = self._find_last_complete_paragraph(lines)
        if continuation:
            return continuation

        # Strategy 4: Use original content for reference
        if original_content:
            continuation = self._find_divergence_point(content, original_content)
            if continuation:
                return continuation

        return None

    def _find_last_complete_section(self, lines: List[str]) -> Optional[str]:
        """Find the last complete section (header with content)."""
        for i in range(len(lines) - 1, -1, -1):
            line = lines[i].strip()
            if line.startswith("## ") and i + 1 < len(lines):
                # Check if there's content after this header
                next_lines = []
                for j in range(i + 1, min(i + 10, len(lines))):
                    if lines[j].strip() and not lines[j].strip().startswith("##"):
                        next_lines.append(lines[j])
                    else:
                        break

                if next_lines:
                    return "\n".join(lines[i:])
        return None

    def _find_last_complete_code_block(self, lines: List[str]) -> Optional[str]:
        """Find the last complete code block."""
        in_code_block = False
        code_block_start = -1

        for i in range(len(lines) - 1, -1, -1):
            line = lines[i].strip()
            if line.startswith("```") and not in_code_block:
                in_code_block = True
                code_block_start = i
            elif line.startswith("```") and in_code_block:
                return "\n".join(lines[code_block_start:])
        return None

    def _find_last_complete_paragraph(self, lines: List[str]) -> Optional[str]:
        """Find the last complete paragraph (ends with period)."""
        for i in range(len(lines) - 1, -1, -1):
            line = lines[i].strip()
            if (
                line
                and line.endswith(".")
                and not line.startswith("#")
                and not line.startswith("```")
            ):
                return "\n".join(lines[i:])
        return None

    def _find_divergence_point(
        self, content: str, original_content: str
    ) -> Optional[str]:
        """Find where generated content diverges from original."""
        min_len = min(len(content), len(original_content))
        common_length = 0

        for i in range(1, min_len + 1):
            if content[-i:] == original_content[-i:]:
                common_length = i
            else:
                break

        if common_length > 100:
            return content[-(common_length + 100) :]
        return None

    def check_completeness(
        self, content: str, target_file: str, original_content: Optional[str] = None
    ) -> CompletenessResult:
        """
        Check if content appears to be complete.

        CRITICAL: If original_content is provided, generated content MUST be
        at least 90% of original length to be considered complete.

        Args:
            content: Generated content to check
            target_file: Target file path for context
            original_content: Original content for length comparison

        Returns:
            CompletenessResult with completeness details
        """
        if not content or len(content.strip()) < self.MIN_CONTENT_LENGTH:
            return CompletenessResult(
                is_complete=False, reason="Content too short", confidence=0.0
            )

        # Critical: Check length ratio first
        if original_content and isinstance(original_content, str):
            generated_len = len(content)
            original_len = len(original_content)

            if generated_len < original_len * self.COMPLETION_RATIO:
                return CompletenessResult(
                    is_complete=False,
                    reason=f"Length ratio {generated_len / original_len:.1%} < {self.COMPLETION_RATIO:.0%}",
                    confidence=generated_len / original_len,
                )

        # Check balanced code blocks
        code_block_count = content.count("```")
        if code_block_count > 0 and code_block_count % 2 != 0:
            return CompletenessResult(
                is_complete=False, reason="Unbalanced code blocks", confidence=0.3
            )

        # File-type specific checks
        file_type = self._detect_file_type(target_file)

        if file_type == FileType.RMARKDOWN:
            result = self._check_rmarkdown_completeness(content, code_block_count)
            if result.is_complete:
                return result

        if file_type == FileType.MARKDOWN:
            result = self._check_markdown_completeness(content)
            if result.is_complete:
                return result

        if file_type == FileType.PYTHON:
            result = self._check_python_completeness(content)
            if result is not None:
                return result

        if file_type == FileType.JAVASCRIPT:
            result = self._check_javascript_completeness(content)
            if result is not None:
                return result

        # Generic checks
        if len(content) > self.REASONABLE_LENGTH:
            result = self._check_generic_completeness(content)
            if result.is_complete:
                return result

        return CompletenessResult(
            is_complete=False, reason="No completion markers found", confidence=0.5
        )

    def _check_rmarkdown_completeness(
        self, content: str, code_block_count: int
    ) -> CompletenessResult:
        """Check RMarkdown-specific completeness."""
        if not content.startswith("---"):
            return CompletenessResult(
                is_complete=False, reason="Missing YAML frontmatter", confidence=0.2
            )

        conclusion_patterns = [
            "sessionInfo()",
            "session.info()",
            "## Conclusion",
            "## Summary",
            "## Session Info",
            "</details>",
            "knitr::knit(",
        ]

        content_lower = content.lower()
        has_conclusion = any(p.lower() in content_lower for p in conclusion_patterns)

        if has_conclusion and code_block_count > 0:
            return CompletenessResult(
                is_complete=True,
                reason="Has conclusion markers and code blocks",
                confidence=0.9,
            )

        return CompletenessResult(is_complete=False, confidence=0.5)

    def _check_markdown_completeness(self, content: str) -> CompletenessResult:
        """Check Markdown-specific completeness."""
        conclusion_patterns = [
            "## Conclusion",
            "## Summary",
            "## Next Steps",
            "## Further Reading",
            "## References",
            "## License",
        ]

        content_lower = content.lower()
        has_conclusion = any(p.lower() in content_lower for p in conclusion_patterns)

        if has_conclusion and len(content) > 2000:
            return CompletenessResult(
                is_complete=True,
                reason="Has conclusion section and reasonable length",
                confidence=0.85,
            )

        return CompletenessResult(is_complete=False, confidence=0.5)

    def _check_python_completeness(self, content: str) -> Optional[CompletenessResult]:
        """Check Python-specific completeness."""
        # Check balanced brackets
        if (
            content.count("(") != content.count(")")
            or content.count("[") != content.count("]")
            or content.count("{") != content.count("}")
        ):
            return CompletenessResult(
                is_complete=False, reason="Unbalanced brackets", confidence=0.2
            )

        lines = [line for line in content.split("\n") if line.strip()]
        if len(lines) > 20:
            last_line = lines[-1].strip()
            if not (
                last_line.endswith(":")
                or last_line.endswith("\\")
                or last_line.endswith(",")
            ):
                return CompletenessResult(
                    is_complete=True,
                    reason="Balanced structure and proper ending",
                    confidence=0.8,
                )
        return None

    def _check_javascript_completeness(
        self, content: str
    ) -> Optional[CompletenessResult]:
        """Check JavaScript-specific completeness."""
        if content.count("{") != content.count("}") or content.count(
            "("
        ) != content.count(")"):
            return CompletenessResult(
                is_complete=False, reason="Unbalanced brackets", confidence=0.2
            )

        lines = [line for line in content.split("\n") if line.strip()]
        if len(lines) > 20:
            last_line = lines[-1].strip()
            if (
                last_line.endswith("}")
                or last_line.endswith(";")
                or last_line.endswith("*/")
                or last_line.startswith("//")
            ):
                return CompletenessResult(
                    is_complete=True,
                    reason="Balanced structure and proper ending",
                    confidence=0.8,
                )
        return None

    def _check_generic_completeness(self, content: str) -> CompletenessResult:
        """Check generic completeness patterns."""
        lines = content.split("\n")
        last_lines = [line.strip() for line in lines[-10:] if line.strip()]

        if last_lines:
            last_line = last_lines[-1]
            complete_endings = [".", "```", "---", "</details>", "}", ";", "*/"]

            if any(last_line.endswith(ending) for ending in complete_endings):
                return CompletenessResult(
                    is_complete=True,
                    reason="Ends with completion marker",
                    confidence=0.7,
                )

        return CompletenessResult(is_complete=False, confidence=0.4)

    def is_truncated(
        self, content: str, target_file: str, original_content: Optional[str] = None
    ) -> bool:
        """
        Simple boolean check for truncation.

        Convenience method that wraps detect_truncation().
        """
        result = self.detect_truncation(content, target_file, original_content)
        return result.is_truncated

    def is_complete(
        self, content: str, target_file: str, original_content: Optional[str] = None
    ) -> bool:
        """
        Simple boolean check for completeness.

        Convenience method that wraps check_completeness().
        """
        result = self.check_completeness(content, target_file, original_content)
        return result.is_complete
