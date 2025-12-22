"""
Enhanced Generation Test Manager (V2).

Provides multi-file error injection testing with comprehensive tracking.
Refactored to use shared utilities from base_test_manager and file_selection.
"""

from __future__ import annotations

import os
import json
import shutil
from typing import Dict, List, Optional, Any

from bioguider.managers.base_test_manager import BaseTestManager, InjectionResult
from bioguider.managers.generation_manager import DocumentationGenerationManager
from bioguider.managers.config import TestConfig
from bioguider.generation.unified_metrics import (
    UnifiedMetricsEvaluator,
    EvaluationResult,
)
from bioguider.agents.agent_utils import read_file, write_file


class GenerationTestManagerV2(BaseTestManager):
    """
    Enhanced test manager with multi-file error injection.

    Features:
    - Injects errors into ALL files in multiple categories
    - Comprehensive per-file and per-category statistics
    - Detailed reporting (fixed vs unchanged)
    - Saves corrupted, original, and fixed versions for audit

    This is a refactored version that uses shared utilities from:
    - BaseTestManager: Common injection and file handling
    - FileSelector: File selection logic
    - UnifiedMetricsEvaluator: Evaluation metrics
    """

    def __init__(
        self,
        llm,
        step_callback,
        config: Optional[TestConfig] = None,
    ):
        """
        Initialize the test manager.

        Args:
            llm: Language model for injection and fixing
            step_callback: Callback for progress reporting
            config: Optional test configuration
        """
        super().__init__(llm, step_callback)
        self.config = config or TestConfig()

    def run_quant_test(
        self,
        report_path: str,
        baseline_repo_path: str,
        tmp_repo_path: str,
        min_per_category: int = 3,
    ) -> str:
        """
        Run quantifiable testing with multi-file error injection.

        Args:
            report_path: Path to evaluation report JSON
            baseline_repo_path: Path to baseline repository
            tmp_repo_path: Path for temporary test repository
            min_per_category: Minimum errors per category to inject

        Returns:
            Path to the output directory
        """
        # 1. Select target files
        self.print_step("SelectFiles", "Identifying target files...")
        file_selection = self.select_target_files(
            baseline_repo_path, max_per_category=self.config.max_files_per_category
        )

        # 2. Prepare temporary repository
        self.prepare_tmp_repo(baseline_repo_path, tmp_repo_path)

        # 3. Extract project terms for targeted injection
        project_terms = self.extract_project_terms(tmp_repo_path)

        # 4. Inject errors into files
        self.print_step(
            "InjectErrors", f"Injecting {min_per_category} errors per category..."
        )
        injection_results = self.inject_errors_into_files(
            file_selection=file_selection,
            tmp_repo_path=tmp_repo_path,
            min_per_category=min_per_category,
            project_terms=project_terms,
        )

        # 5. Save injection manifest
        inj_path = self.save_injection_manifest(injection_results, tmp_repo_path)

        # 6. Run generation/fixing
        self.print_step("RunGeneration", "Running BioGuider to fix errors...")
        gen = DocumentationGenerationManager(self.llm, self.step_callback)
        out_dir = gen.run(report_path=report_path, repo_path=tmp_repo_path)

        # 7. Evaluate fixes
        self.print_step("EvaluateFixes", "Evaluating error corrections...")
        manifests = self.convert_injections_to_manifests(injection_results)
        evaluator = UnifiedMetricsEvaluator(
            llm=None,  # No semantic FP detection for V2
            detect_fp=self.config.detect_semantic_fp,
        )
        results = evaluator.evaluate_multiple_files(manifests, out_dir)

        # 8. Save results
        results_path = os.path.join(out_dir, "GEN_TEST_RESULTS.json")
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(results.to_dict(), f, indent=2)

        # Copy injection manifest to output
        shutil.copy(inj_path, os.path.join(out_dir, "INJECTION_MANIFEST.json"))

        # 9. Generate report
        level = self._determine_level(min_per_category)
        self._generate_report(results, out_dir, level)

        # 10. Save versioned files
        self._save_versioned_files(injection_results, out_dir)

        self.print_step("TestComplete", f"Results saved to {out_dir}")
        return out_dir

    def run_quant_suite(
        self,
        report_path: str,
        baseline_repo_path: str,
        base_tmp_repo_path: str,
        levels: Dict[str, int],
    ) -> Dict[str, str]:
        """
        Run test suite across multiple error levels.

        Args:
            report_path: Path to evaluation report JSON
            baseline_repo_path: Path to baseline repository
            base_tmp_repo_path: Base path for temporary repositories
            levels: Dict mapping level names to error counts

        Returns:
            Dict mapping level names to output directories
        """
        results = {}
        for level, min_cnt in levels.items():
            self.print_step(
                f"RunLevel:{level.upper()}", f"Running with {min_cnt} errors per file"
            )
            tmp_repo_path = f"{base_tmp_repo_path}_{level}"
            out_dir = self.run_quant_test(
                report_path, baseline_repo_path, tmp_repo_path, min_per_category=min_cnt
            )
            results[level] = out_dir
        return results

    def _determine_level(self, min_per_category: int) -> str:
        """Determine test level based on error count."""
        if min_per_category <= 3:
            return "low"
        elif min_per_category <= 7:
            return "mid"
        else:
            return "high"

    def _generate_report(
        self,
        results: EvaluationResult,
        output_dir: str,
        level: str,
    ):
        """Generate a comprehensive markdown report."""
        lines = [
            "# BioGuider Quantifiable Testing Results\n",
            f"**Test Level**: {level.upper()}\n",
            "\n---\n",
            "\n## Executive Summary\n",
            f"\n### Overall Performance\n",
            f"- **Success Rate**: {results.success_rate}%\n",
            f"- **Total Files Tested**: {results.total_files}\n",
            f"- **Total Errors Injected**: {results.total_errors}\n",
            f"- **Errors Fixed**: {results.true_positives} "
            f"({round(results.fix_rate * 100, 1)}%)\n",
            f"- **Errors Unchanged**: {results.false_negatives} "
            f"({round((1 - results.fix_rate) * 100, 1) if results.total_errors > 0 else 0}%)\n",
            "\n---\n",
        ]

        # Performance by file
        lines.append("\n## Performance by File\n")
        for file_path, metrics in results.per_file.items():
            fix_rate = (
                metrics.true_positives / metrics.total_errors * 100
                if metrics.total_errors > 0
                else 0
            )
            lines.append(f"\n### `{file_path}`\n")
            lines.append(f"- Category: {metrics.category}\n")
            lines.append(f"- Errors Injected: {metrics.total_errors}\n")
            lines.append(
                f"- Errors Fixed: {metrics.true_positives} ({fix_rate:.1f}%)\n"
            )
            lines.append(f"- Errors Unchanged: {metrics.false_negatives}\n")

        # Performance by error category
        lines.append("\n---\n")
        lines.append("\n## Performance by Error Category\n")
        lines.append("\n| Category | Total | Fixed | Unchanged | Fix Rate |\n")
        lines.append("|----------|-------|-------|-----------|----------|\n")

        for cat, metrics in sorted(
            results.per_category.items(), key=lambda x: -x[1].total
        ):
            fix_rate = metrics.fix_rate * 100
            lines.append(
                f"| {cat} | {metrics.total} | {metrics.fixed} | "
                f"{metrics.unchanged} | {fix_rate:.1f}% |\n"
            )

        # Notes
        lines.append("\n---\n")
        lines.append("\n## Notes\n")
        lines.append("- Original, corrupted, and fixed versions saved for each file\n")
        lines.append(
            "- Detailed injection manifests available in `INJECTION_MANIFEST.json`\n"
        )
        lines.append("- Complete results data in `GEN_TEST_RESULTS.json`\n")

        report_path = os.path.join(output_dir, "GEN_TEST_REPORT.md")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("".join(lines))

    def _save_versioned_files(
        self,
        injection_results: Dict[str, InjectionResult],
        output_dir: str,
    ):
        """Save original and corrupted versions for audit."""
        for rel_path, result in injection_results.items():
            base_name = os.path.basename(rel_path)
            base_dir = os.path.dirname(rel_path)

            # Parse file extension
            if "." in base_name:
                name_parts = base_name.rsplit(".", 1)
                base_name_no_ext = name_parts[0]
                ext = "." + name_parts[1]
            else:
                base_name_no_ext = base_name
                ext = ""

            # Create versioned filenames
            orig_name = f"{base_name_no_ext}.original{ext}"
            corr_name = f"{base_name_no_ext}.corrupted{ext}"

            # Determine save directory
            if base_name == "README.md":
                save_dir = output_dir
            else:
                save_dir = (
                    os.path.join(output_dir, base_dir) if base_dir else output_dir
                )

            os.makedirs(save_dir, exist_ok=True)

            # Save files
            if self.config.save_original:
                write_file(os.path.join(save_dir, orig_name), result.baseline_content)
            if self.config.save_corrupted:
                write_file(os.path.join(save_dir, corr_name), result.corrupted_content)
