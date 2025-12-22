"""
Base class for test managers that handle error injection testing.

Provides common interface and shared functionality for:
- GenerationTestManager
- GenerationTestManagerV2
- BenchmarkManager
"""

from __future__ import annotations

import os
import json
import shutil
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

from bioguider.generation.llm_injector import LLMErrorInjector
from bioguider.managers.file_selection import (
    FileSelector,
    ProjectTermExtractor,
    FileSelectionResult,
)
from bioguider.managers.config import TestConfig, BenchmarkConfig


@dataclass
class InjectionResult:
    """Result of error injection into a file."""

    rel_path: str
    category: str
    original_path: str
    corrupted_path: str
    manifest: Dict[str, Any]
    baseline_content: str
    corrupted_content: str
    error_count: int = 0

    def __post_init__(self):
        if self.error_count == 0:
            self.error_count = len(self.manifest.get("errors", []))


@dataclass
class TestResult:
    """Result of a test run."""

    output_dir: str
    total_files: int
    total_errors: int
    success_rate: float
    results_path: str
    report_path: str
    duration_seconds: float = 0.0


class BaseTestManager(ABC):
    """
    Abstract base class for test managers.

    Provides common functionality:
    - Step callback handling
    - File selection via FileSelector
    - Term extraction via ProjectTermExtractor
    - Error injection via LLMErrorInjector
    """

    def __init__(
        self,
        llm: Any,
        step_callback: Optional[Callable[[str, str], None]] = None,
    ):
        """
        Initialize the test manager.

        Args:
            llm: Language model for error injection and fixing
            step_callback: Optional callback for progress reporting
        """
        self.llm = llm
        self.step_callback = step_callback
        self.injector = LLMErrorInjector(llm)

    def print_step(self, name: str, output: str = ""):
        """Report progress via callback or print."""
        if self.step_callback:
            self.step_callback(step_name=name, step_output=output)
        else:
            print(f"[{name}] {output}")

    # =========================================================================
    # File Selection (Delegated to FileSelector)
    # =========================================================================

    def select_target_files(
        self,
        repo_path: str,
        max_per_category: Optional[int] = None,
    ) -> FileSelectionResult:
        """
        Select target files for error injection.

        Args:
            repo_path: Path to the repository
            max_per_category: Optional limit on files per category

        Returns:
            FileSelectionResult with selected files
        """
        selector = FileSelector(repo_path)
        result = selector.select_all()

        if max_per_category:
            result = result.limit_per_category(max_per_category)

        self.print_step(
            "FilesSelected",
            f"{result.total_files} files across {len(result.files_by_category)} categories",
        )

        return result

    def extract_project_terms(
        self,
        repo_path: str,
        max_terms: int = 20,
    ) -> List[str]:
        """
        Extract function names and key terms from the codebase.

        Args:
            repo_path: Path to the repository
            max_terms: Maximum number of terms to return

        Returns:
            List of extracted terms
        """
        extractor = ProjectTermExtractor(repo_path)
        terms = extractor.extract(max_terms=max_terms)

        self.print_step(
            "ExtractTerms",
            f"Found {len(terms)} project terms: {', '.join(terms[:5])}...",
        )

        return terms

    # =========================================================================
    # Repository Preparation
    # =========================================================================

    def prepare_tmp_repo(
        self,
        baseline_repo_path: str,
        tmp_repo_path: str,
        clean_existing: bool = True,
    ) -> str:
        """
        Prepare a temporary copy of the repository for testing.

        Args:
            baseline_repo_path: Path to the original repository
            tmp_repo_path: Path for the temporary copy
            clean_existing: Whether to remove existing tmp directory

        Returns:
            Path to the temporary repository
        """
        if clean_existing and os.path.exists(tmp_repo_path):
            shutil.rmtree(tmp_repo_path)

        shutil.copytree(
            baseline_repo_path,
            tmp_repo_path,
            symlinks=False,
            ignore=shutil.ignore_patterns(".git", "__pycache__", "*.pyc"),
        )

        self.print_step("PrepareRepo", f"Copied to {tmp_repo_path}")

        return tmp_repo_path

    # =========================================================================
    # Error Injection
    # =========================================================================

    def inject_errors_into_file(
        self,
        fpath: str,
        category: str,
        tmp_repo_path: str,
        min_per_category: int,
        project_terms: Optional[List[str]] = None,
    ) -> Optional[InjectionResult]:
        """
        Inject errors into a single file.

        Args:
            fpath: Path to the file
            category: Category of the file (readme, tutorial, etc.)
            tmp_repo_path: Path to the temporary repository
            min_per_category: Minimum errors per category
            project_terms: Optional list of project-specific terms

        Returns:
            InjectionResult or None if injection failed
        """
        from bioguider.agents.agent_utils import read_file, write_file

        if not os.path.exists(fpath):
            return None

        baseline_content = read_file(fpath) or ""
        if not baseline_content.strip():
            return None

        try:
            # Inject errors
            corrupted, manifest = self.injector.inject(
                baseline_content,
                min_per_category=min_per_category,
                project_terms=project_terms or [],
            )

            # Determine relative path
            rel_path = self._get_relative_path(fpath, tmp_repo_path)

            # Write corrupted file
            corrupted_path = os.path.join(tmp_repo_path, rel_path)
            os.makedirs(os.path.dirname(corrupted_path), exist_ok=True)
            write_file(corrupted_path, corrupted)

            # Add file path to each error for tracking
            for error in manifest.get("errors", []):
                error["file_path"] = rel_path

            return InjectionResult(
                rel_path=rel_path,
                category=category,
                original_path=fpath,
                corrupted_path=corrupted_path,
                manifest=manifest,
                baseline_content=baseline_content,
                corrupted_content=corrupted,
            )

        except Exception as e:
            self.print_step(f"InjectionError:{os.path.basename(fpath)}", str(e))
            return None

    def inject_errors_into_files(
        self,
        file_selection: FileSelectionResult,
        tmp_repo_path: str,
        min_per_category: int,
        project_terms: Optional[List[str]] = None,
    ) -> Dict[str, InjectionResult]:
        """
        Inject errors into multiple files.

        Args:
            file_selection: Selected files by category
            tmp_repo_path: Path to the temporary repository
            min_per_category: Minimum errors per category
            project_terms: Optional list of project-specific terms

        Returns:
            Dict mapping relative paths to InjectionResults
        """
        all_results: Dict[str, InjectionResult] = {}

        for category, files in file_selection.files_by_category.items():
            self.print_step(
                f"InjectErrors:{category.title()}",
                f"Injecting {min_per_category} errors per file into {len(files)} files",
            )

            for fpath in files:
                result = self.inject_errors_into_file(
                    fpath=fpath,
                    category=category,
                    tmp_repo_path=tmp_repo_path,
                    min_per_category=min_per_category,
                    project_terms=project_terms,
                )

                if result:
                    all_results[result.rel_path] = result
                    self.print_step(
                        f"Injected:{os.path.basename(fpath)}",
                        f"{result.error_count} errors",
                    )

        total_errors = sum(r.error_count for r in all_results.values())
        self.print_step(
            "InjectionComplete", f"{total_errors} errors in {len(all_results)} files"
        )

        return all_results

    # =========================================================================
    # Manifest Handling
    # =========================================================================

    def save_injection_manifest(
        self,
        injection_results: Dict[str, InjectionResult],
        output_dir: str,
        filename: str = "INJECTION_MANIFEST.json",
    ) -> str:
        """
        Save injection manifest to JSON.

        Args:
            injection_results: Dict of injection results
            output_dir: Directory to save to
            filename: Name of the manifest file

        Returns:
            Path to the saved manifest
        """
        all_errors = []
        files_info = {}

        for rel_path, result in injection_results.items():
            file_errors = result.manifest.get("errors", [])
            files_info[rel_path] = {
                "category": result.category,
                "original_path": result.original_path,
                "corrupted_path": result.corrupted_path,
                "error_count": result.error_count,
                "errors": file_errors,
            }
            all_errors.extend(file_errors)

        manifest = {
            "total_files": len(injection_results),
            "total_errors": len(all_errors),
            "files": files_info,
            "errors": all_errors,
        }

        manifest_path = os.path.join(output_dir, filename)
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)

        return manifest_path

    def load_injection_manifest(
        self,
        manifest_path: str,
    ) -> Dict[str, Any]:
        """
        Load injection manifest from JSON.

        Args:
            manifest_path: Path to the manifest file

        Returns:
            Manifest dict
        """
        with open(manifest_path, "r", encoding="utf-8") as f:
            return json.load(f)

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _get_relative_path(self, fpath: str, base_path: str) -> str:
        """Get relative path from base, handling edge cases."""
        try:
            rel = os.path.relpath(fpath, base_path)
            if rel.startswith("../"):
                # File is outside base path, use basename
                return os.path.basename(fpath)
            return rel
        except ValueError:
            # Different drives on Windows
            return os.path.basename(fpath)

    def convert_injections_to_manifests(
        self,
        injection_results: Dict[str, InjectionResult],
    ) -> Dict[str, Dict[str, Any]]:
        """
        Convert InjectionResults to the manifest format expected by evaluators.

        Args:
            injection_results: Dict of injection results

        Returns:
            Dict in the format expected by UnifiedMetricsEvaluator
        """
        return {
            rel_path: {
                "category": result.category,
                "manifest": result.manifest,
                "baseline_content": result.baseline_content,
                "corrupted_content": result.corrupted_content,
            }
            for rel_path, result in injection_results.items()
        }

    # =========================================================================
    # Abstract Methods
    # =========================================================================

    @abstractmethod
    def run_quant_test(
        self,
        report_path: str,
        baseline_repo_path: str,
        tmp_repo_path: str,
        min_per_category: int = 3,
    ) -> str:
        """
        Run a quantifiable test.

        Args:
            report_path: Path to evaluation report JSON
            baseline_repo_path: Path to the baseline repository
            tmp_repo_path: Path for the temporary test repository
            min_per_category: Minimum errors per category

        Returns:
            Path to the output directory
        """
        pass

    @abstractmethod
    def run_quant_suite(
        self,
        report_path: str,
        baseline_repo_path: str,
        base_tmp_repo_path: str,
        levels: Dict[str, int],
    ) -> Dict[str, str]:
        """
        Run a suite of tests at different levels.

        Args:
            report_path: Path to evaluation report JSON
            baseline_repo_path: Path to the baseline repository
            base_tmp_repo_path: Base path for temporary repositories
            levels: Dict mapping level names to error counts

        Returns:
            Dict mapping level names to output directories
        """
        pass
