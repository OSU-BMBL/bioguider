"""
Prompt template management for LLM content generation.

This module provides:
- PromptLoader: Load and format prompt templates from files
- Pre-defined prompt template names
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Dict, Any


class PromptTemplate:
    """Named constants for prompt templates."""

    SECTION = "section"
    FULL_DOCUMENT = "full_document"
    README_COMPREHENSIVE = "readme_comprehensive"
    CONTINUATION = "continuation"
    RMARKDOWN_CHUNK = "rmarkdown_chunk"


class PromptLoader:
    """
    Loads and formats prompt templates from .txt files.

    Templates use Python string formatting with named placeholders.
    Example: "Hello {name}, you have {count} messages."
    """

    def __init__(self, prompts_dir: Optional[str] = None):
        """
        Initialize the prompt loader.

        Args:
            prompts_dir: Directory containing .txt prompt files.
                        Defaults to the 'prompts' directory next to this file.
        """
        if prompts_dir is None:
            self.prompts_dir = Path(__file__).parent
        else:
            self.prompts_dir = Path(prompts_dir)

        self._cache: Dict[str, str] = {}

    def load(self, name: str) -> str:
        """
        Load a prompt template by name.

        Args:
            name: Template name (without .txt extension)

        Returns:
            Template string with placeholders

        Raises:
            FileNotFoundError: If template file doesn't exist
        """
        if name in self._cache:
            return self._cache[name]

        template_path = self.prompts_dir / f"{name}.txt"
        if not template_path.exists():
            raise FileNotFoundError(f"Prompt template not found: {template_path}")

        template = template_path.read_text(encoding="utf-8")
        self._cache[name] = template
        return template

    def format(self, name: str, **kwargs: Any) -> str:
        """
        Load and format a prompt template.

        Args:
            name: Template name
            **kwargs: Values for template placeholders

        Returns:
            Formatted prompt string
        """
        template = self.load(name)
        return template.format(**kwargs)

    def get_available_templates(self) -> list[str]:
        """List all available template names."""
        return [f.stem for f in self.prompts_dir.glob("*.txt")]

    def clear_cache(self):
        """Clear the template cache."""
        self._cache.clear()


# Default loader instance
_default_loader: Optional[PromptLoader] = None


def get_prompt_loader() -> PromptLoader:
    """Get the default prompt loader instance."""
    global _default_loader
    if _default_loader is None:
        _default_loader = PromptLoader()
    return _default_loader


def load_prompt(name: str) -> str:
    """Convenience function to load a prompt template."""
    return get_prompt_loader().load(name)


def format_prompt(name: str, **kwargs: Any) -> str:
    """Convenience function to load and format a prompt template."""
    return get_prompt_loader().format(name, **kwargs)


__all__ = [
    "PromptLoader",
    "PromptTemplate",
    "get_prompt_loader",
    "load_prompt",
    "format_prompt",
]
