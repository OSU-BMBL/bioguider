"""
Shared file selection utilities for test managers and benchmarking.

This module consolidates file selection logic that was previously duplicated
across GenerationTestManagerV2 and BenchmarkManager.
"""

from __future__ import annotations

import os
import re
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

from bioguider.agents.agent_utils import read_file


class FileCategory(str, Enum):
    """Categories of documentation files for processing."""

    README = "readme"
    TUTORIAL = "tutorial"
    USERGUIDE = "userguide"
    INSTALLATION = "installation"
    RD_DOC = "rd_doc"


@dataclass
class FileSelectionResult:
    """Result of file selection operation."""

    files_by_category: Dict[str, List[str]] = field(default_factory=dict)
    total_files: int = 0

    def __post_init__(self):
        if not self.files_by_category:
            self.files_by_category = {cat.value: [] for cat in FileCategory}

    def get_all_files(self) -> List[str]:
        """Get flat list of all selected files."""
        return [f for files in self.files_by_category.values() for f in files]

    def limit_per_category(self, max_files: int) -> "FileSelectionResult":
        """Return new result with limited files per category."""
        limited = {
            cat: files[:max_files] for cat, files in self.files_by_category.items()
        }
        return FileSelectionResult(
            files_by_category=limited, total_files=sum(len(f) for f in limited.values())
        )


class FileSelector:
    """
    Selects target files for error injection and documentation processing.

    Consolidates logic previously duplicated in:
    - GenerationTestManagerV2._select_target_files()
    - BenchmarkManager._select_target_files()
    """

    # Common file patterns for each category
    README_PATTERNS = ["README.md", "README.rst", "README.txt", "readme.md"]
    INSTALL_PATTERNS = ["install", "INSTALL", "installation", "INSTALLATION"]
    INSTALL_EXTENSIONS = [".md", ".Rmd", ".rst", ".txt"]
    DOC_EXTENSIONS = [".md", ".rst", ".Rmd", ".Rd", ".txt"]

    def __init__(self, repo_path: str):
        """
        Initialize file selector.

        Args:
            repo_path: Path to the repository root
        """
        self.repo_path = repo_path

    def select_all(
        self, include_categories: Optional[List[FileCategory]] = None
    ) -> FileSelectionResult:
        """
        Select target files across all or specified categories.

        Args:
            include_categories: Optional list of categories to include.
                              If None, includes all categories.

        Returns:
            FileSelectionResult with files organized by category
        """
        if include_categories is None:
            include_categories = list(FileCategory)

        result = FileSelectionResult()

        for category in include_categories:
            if category == FileCategory.README:
                result.files_by_category[category.value] = self._find_readme_files()
            elif category == FileCategory.TUTORIAL:
                result.files_by_category[category.value] = self._find_tutorial_files()
            elif category == FileCategory.USERGUIDE:
                result.files_by_category[category.value] = self._find_userguide_files()
            elif category == FileCategory.INSTALLATION:
                result.files_by_category[category.value] = (
                    self._find_installation_files()
                )
            elif category == FileCategory.RD_DOC:
                result.files_by_category[category.value] = self._find_rd_doc_files()

        result.total_files = sum(
            len(files) for files in result.files_by_category.values()
        )
        return result

    def _find_readme_files(self) -> List[str]:
        """Find README files in the repository root."""
        readme_files = []
        for pattern in self.README_PATTERNS:
            readme_path = os.path.join(self.repo_path, pattern)
            if os.path.exists(readme_path):
                readme_files.append(readme_path)
        return readme_files

    def _find_tutorial_files(self) -> List[str]:
        """Find tutorial files (typically RMarkdown vignettes)."""
        tutorial_files = []

        # Check vignettes directory (common in R packages)
        vignettes_dir = os.path.join(self.repo_path, "vignettes")
        if os.path.isdir(vignettes_dir):
            for f in sorted(os.listdir(vignettes_dir)):
                if f.endswith(".Rmd") and not f.startswith("."):
                    # Exclude installation-specific vignettes
                    if "install" not in f.lower():
                        tutorial_files.append(os.path.join(vignettes_dir, f))

        # Check tutorials directory
        tutorials_dir = os.path.join(self.repo_path, "tutorials")
        if os.path.isdir(tutorials_dir):
            for f in sorted(os.listdir(tutorials_dir)):
                if any(f.endswith(ext) for ext in self.DOC_EXTENSIONS):
                    if not f.startswith("."):
                        tutorial_files.append(os.path.join(tutorials_dir, f))

        return tutorial_files

    def _find_userguide_files(self) -> List[str]:
        """Find user guide files (typically in docs directory)."""
        userguide_files = []

        docs_dir = os.path.join(self.repo_path, "docs")
        if os.path.isdir(docs_dir):
            for f in sorted(os.listdir(docs_dir)):
                if f.endswith(".md") and not f.startswith("."):
                    userguide_files.append(os.path.join(docs_dir, f))

        # Also check for standalone guide files
        for name in ["USERGUIDE.md", "USER_GUIDE.md", "GUIDE.md", "USAGE.md"]:
            guide_path = os.path.join(self.repo_path, name)
            if os.path.exists(guide_path):
                userguide_files.append(guide_path)

        return userguide_files

    def _find_installation_files(self) -> List[str]:
        """Find installation documentation files."""
        install_files = []

        # Check root directory for install files
        for pattern in self.INSTALL_PATTERNS:
            for ext in self.INSTALL_EXTENSIONS:
                fpath = os.path.join(self.repo_path, pattern + ext)
                if os.path.exists(fpath):
                    install_files.append(fpath)

        # Check vignettes for installation guides
        vignettes_dir = os.path.join(self.repo_path, "vignettes")
        if os.path.isdir(vignettes_dir):
            for f in os.listdir(vignettes_dir):
                if "install" in f.lower():
                    if f.endswith(".Rmd") or f.endswith(".md"):
                        fpath = os.path.join(vignettes_dir, f)
                        if fpath not in install_files:
                            install_files.append(fpath)

        return install_files

    def _find_rd_doc_files(self, max_files: int = 20) -> List[str]:
        """Find R documentation files in man/ directory, capped at max_files.

        Skips auto-generated stubs (files whose only non-comment content is
        \\name + \\alias + \\title with no \\description or \\arguments).
        """
        man_dir = os.path.join(self.repo_path, "man")
        if not os.path.isdir(man_dir):
            return []

        rd_files = []
        for f in sorted(os.listdir(man_dir)):
            if not f.endswith(".Rd") or f.startswith("."):
                continue
            fpath = os.path.join(man_dir, f)
            try:
                content = open(fpath, encoding="utf-8", errors="replace").read()
            except OSError:
                continue
            # Skip stubs — must have at least \description{} and \arguments{}
            if r"\description{" not in content or r"\arguments{" not in content:
                continue
            rd_files.append(fpath)
            if len(rd_files) >= max_files:
                break

        return rd_files


class ProjectTermExtractor:
    """
    Extracts function names and key terms from a codebase.

    Consolidates logic previously duplicated in:
    - GenerationTestManagerV2._extract_project_terms()
    - BenchmarkManager._extract_project_terms()
    """

    # Terms to exclude from extraction
    EXCLUDED_TERMS = frozenset(
        {
            "init",
            "self",
            "setup",
            "test",
            "main",
            "true",
            "false",
            "none",
            "return",
            "import",
            "from",
            "class",
            "function",
        }
    )

    # Minimum term length to include
    MIN_TERM_LENGTH = 5

    # Maximum terms to return
    MAX_TERMS = 20

    def __init__(self, repo_path: str):
        """
        Initialize term extractor.

        Args:
            repo_path: Path to the repository root
        """
        self.repo_path = repo_path

    def extract(self, max_terms: Optional[int] = None) -> List[str]:
        """
        Extract function names and key terms from the codebase.

        Args:
            max_terms: Maximum number of terms to return (default: 20)

        Returns:
            List of extracted terms, sorted by frequency
        """
        if max_terms is None:
            max_terms = self.MAX_TERMS

        terms = Counter()

        for root, _, files in os.walk(self.repo_path):
            # Skip hidden and cache directories
            if self._should_skip_directory(root):
                continue

            for file in files:
                fpath = os.path.join(root, file)
                file_terms = self._extract_from_file(fpath, file)
                terms.update(file_terms)

        # Filter and return top terms
        filtered_terms = [
            term for term, _ in terms.most_common(50) if self._is_valid_term(term)
        ]

        return filtered_terms[:max_terms]

    def _should_skip_directory(self, path: str) -> bool:
        """Check if directory should be skipped."""
        skip_patterns = {".git", "__pycache__", "node_modules", ".venv", "venv", ".tox"}
        return any(pattern in path for pattern in skip_patterns)

    def _extract_from_file(self, fpath: str, filename: str) -> List[str]:
        """Extract terms from a single file."""
        try:
            content = read_file(fpath)
            if not content:
                return []

            terms = []

            if filename.endswith(".py"):
                terms.extend(self._extract_python_terms(content))
            elif filename.endswith(".R"):
                terms.extend(self._extract_r_terms(content))
            elif filename.endswith(".js") or filename.endswith(".ts"):
                terms.extend(self._extract_js_terms(content))

            return terms
        except Exception:
            return []

    def _extract_python_terms(self, content: str) -> List[str]:
        """Extract function and class names from Python code."""
        terms = []
        # Function definitions
        funcs = re.findall(r"def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(", content)
        terms.extend(funcs)
        # Class definitions
        classes = re.findall(r"class\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*[:\(]", content)
        terms.extend(classes)
        return terms

    def _extract_r_terms(self, content: str) -> List[str]:
        """Extract function names from R code."""
        funcs = re.findall(r"([a-zA-Z_.][a-zA-Z0-9_.]*)\s*<-\s*function", content)
        return funcs

    def _extract_js_terms(self, content: str) -> List[str]:
        """Extract function names from JavaScript/TypeScript code."""
        terms = []
        # Function declarations
        funcs = re.findall(r"function\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(", content)
        terms.extend(funcs)
        # Arrow functions assigned to variables
        arrows = re.findall(
            r"(?:const|let|var)\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*(?:\([^)]*\)|[a-zA-Z_][a-zA-Z0-9_]*)\s*=>",
            content,
        )
        terms.extend(arrows)
        return terms

    def _is_valid_term(self, term: str) -> bool:
        """Check if a term should be included."""
        return (
            len(term) >= self.MIN_TERM_LENGTH
            and term.lower() not in self.EXCLUDED_TERMS
        )
