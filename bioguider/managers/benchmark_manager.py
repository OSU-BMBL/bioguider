"""
Benchmark Manager for comprehensive error injection testing.

Provides:
- Stress testing across multiple error count levels
- Multi-process parallel execution
- Multi-model comparison support
- CSV/JSON export of results

Refactored to use shared utilities from base_test_manager and unified_metrics.
"""

from __future__ import annotations

import os
import json
import csv
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable

from bioguider.managers.base_test_manager import BaseTestManager, InjectionResult
from bioguider.managers.generation_manager import DocumentationGenerationManager
from bioguider.managers.config import BenchmarkConfig
from bioguider.generation.unified_metrics import (
    UnifiedMetricsEvaluator,
    EvaluationResult,
)
from bioguider.agents.agent_utils import read_file, write_file


@dataclass
class StressTestResult:
    """Result of a single stress test level."""

    error_count: int
    evaluation_result: EvaluationResult
    output_dir: str
    duration_seconds: float = 0.0


@dataclass
class ModelComparisonResult:
    """Comparison results across multiple models."""

    models: List[str]
    error_count: int
    results: Dict[str, EvaluationResult] = field(default_factory=dict)


class BenchmarkManager(BaseTestManager):
    """
    Manages comprehensive benchmark testing for error injection.

    Features:
    - Stress testing with configurable error levels
    - Multi-process parallel execution for injection
    - Multi-model comparison support
    - Comprehensive result export (JSON, CSV, Markdown)

    Refactored to use shared utilities from:
    - BaseTestManager: Common injection and file handling
    - FileSelector/ProjectTermExtractor: File selection
    - UnifiedMetricsEvaluator: Evaluation with F-score and semantic FP
    """

    def __init__(
        self,
        llm,
        step_callback: Optional[Callable] = None,
        config: Optional[BenchmarkConfig] = None,
    ):
        """
        Initialize the benchmark manager.

        Args:
            llm: Language model for injection and fixing
            step_callback: Optional callback for progress reporting
            config: Optional benchmark configuration
        """
        super().__init__(llm, step_callback)
        self.config = config or BenchmarkConfig()

    # =========================================================================
    # Parallel Injection (Override for performance)
    # =========================================================================

    def inject_errors_parallel(
        self,
        file_selection,
        tmp_repo_path: str,
        min_per_category: int,
        project_terms: Optional[List[str]] = None,
        target_total_errors: Optional[int] = None,
    ) -> Dict[str, InjectionResult]:
        """
        Inject errors into multiple files in parallel.

        Overrides the sequential implementation in BaseTestManager
        for better performance with many files.

        Args:
            target_total_errors: When set, overrides ``min_per_category`` with
                an even-spread derivation so the total injection across all
                files targets roughly this many scorable errors. Used by the
                50/100/200/300 gradient figure. ``min_per_category`` is still
                the lower bound (always ≥1 per eligible slot).
        """
        from bioguider.managers.config import (
            SCORABLE_CATEGORIES,
            min_per_category_from_total,
        )

        all_results: Dict[str, InjectionResult] = {}

        # Flatten file list with categories
        files_with_cats = []
        for category, files in file_selection.files_by_category.items():
            for fpath in files:
                files_with_cats.append((fpath, category))

        # Optional: translate a total-errors budget into a per-category minimum.
        if target_total_errors is not None:
            min_per_category = min_per_category_from_total(
                target_total_errors=target_total_errors,
                n_files=len(files_with_cats),
                n_categories=len(SCORABLE_CATEGORIES),
            )
            self.print_step(
                "BudgetTranslate",
                f"target_total_errors={target_total_errors} -> min_per_category={min_per_category}",
            )

        self.print_step(
            "InjectErrors",
            f"Injecting into {len(files_with_cats)} files with {min_per_category} errors/category (parallel)",
        )

        # Use ThreadPoolExecutor (LLM calls are I/O bound)
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            futures = {}
            for fpath, category in files_with_cats:
                future = executor.submit(
                    self.inject_errors_into_file,
                    fpath,
                    category,
                    tmp_repo_path,
                    min_per_category,
                    project_terms,
                )
                futures[future] = (fpath, category)

            for future in as_completed(futures):
                fpath, category = futures[future]
                try:
                    result = future.result()
                    if result:
                        all_results[result.rel_path] = result
                        self.print_step(
                            f"Injected:{os.path.basename(fpath)}",
                            f"{result.error_count} errors",
                        )
                except Exception as e:
                    self.print_step(
                        f"InjectionFailed:{os.path.basename(fpath)}", str(e)
                    )

        total_errors = sum(r.error_count for r in all_results.values())
        self.print_step(
            "InjectionComplete", f"{total_errors} errors in {len(all_results)} files"
        )

        return all_results

    # =========================================================================
    # Stress Testing
    # =========================================================================

    def run_stress_test(
        self,
        report_path: str,
        baseline_repo_path: str,
        output_base_path: str,
        stress_levels: Optional[List[int]] = None,
    ) -> Dict[int, StressTestResult]:
        """
        Run stress tests across multiple error count levels.

        Args:
            report_path: Path to evaluation report JSON
            baseline_repo_path: Path to baseline repository
            output_base_path: Base path for output directories
            stress_levels: List of error counts to test

        Returns:
            Dict mapping error_count to StressTestResult
        """
        import time

        if stress_levels is None:
            stress_levels = self.config.stress_levels

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        benchmark_dir = os.path.join(output_base_path, f"benchmark_{timestamp}")
        os.makedirs(benchmark_dir, exist_ok=True)

        self.print_step("StressTestStart", f"Testing levels: {stress_levels}")

        # Select target files once
        file_selection = self.select_target_files(
            baseline_repo_path, max_per_category=self.config.max_files_per_category
        )

        results: Dict[int, StressTestResult] = {}

        for level in stress_levels:
            start_time = time.time()
            self.print_step(
                f"StressLevel:{level}", f"Starting with {level} errors per category"
            )

            # Create level-specific directory
            level_dir = os.path.join(benchmark_dir, f"level_{level}")
            tmp_repo_path = os.path.join(level_dir, "tmp_repo")

            # Prepare repository
            self.prepare_tmp_repo(baseline_repo_path, tmp_repo_path)

            # Extract terms
            project_terms = self.extract_project_terms(tmp_repo_path)

            # Inject errors in parallel
            injection_results = self.inject_errors_parallel(
                file_selection, tmp_repo_path, level, project_terms
            )

            # Save injection manifest
            self.save_injection_manifest(
                injection_results, level_dir, "BENCHMARK_MANIFEST.json"
            )

            # Run generation
            injected_files = (
                list(injection_results.keys())
                if self.config.limit_generation_files
                else None
            )
            max_files = len(injected_files) if injected_files else None

            self.print_step(
                "RunGeneration",
                f"Processing {'ONLY ' + str(len(injected_files)) + ' injected' if injected_files else 'ALL'} files...",
            )

            gen = DocumentationGenerationManager(self.llm, self.step_callback)
            out_dir = gen.run(
                report_path=report_path,
                repo_path=tmp_repo_path,
                target_files=injected_files,
                max_files=max_files,
            )

            # Evaluate results
            self.print_step("EvaluateFixes", "Computing benchmark metrics...")
            manifests = self.convert_injections_to_manifests(injection_results)
            evaluator = UnifiedMetricsEvaluator(
                llm=self.llm if self.config.detect_semantic_fp else None,
                detect_fp=self.config.detect_semantic_fp,
            )
            eval_result = evaluator.evaluate_multiple_files(manifests, out_dir)

            duration = time.time() - start_time

            results[level] = StressTestResult(
                error_count=level,
                evaluation_result=eval_result,
                output_dir=level_dir,
                duration_seconds=duration,
            )

            # Save level results
            self._save_level_results(results[level], level_dir)

            self.print_step(
                f"LevelComplete:{level}",
                f"F1={eval_result.f1_score:.3f}, FixRate={eval_result.fix_rate:.3f}",
            )

        # Save aggregate stress test results
        self._save_stress_test_results(results, benchmark_dir)

        self.print_step("StressTestComplete", f"Results saved to {benchmark_dir}")
        return results

    # =========================================================================
    # Total-Error Gradient (F1-vs-error-count figure)
    # =========================================================================

    def run_total_error_gradient(
        self,
        report_path: str,
        baseline_repo_path: str,
        output_base_path: str,
        total_levels: Optional[List[int]] = None,
    ) -> Dict[int, StressTestResult]:
        """
        Run the F1-vs-error-count gradient benchmark.

        Same flow as ``run_stress_test`` but each ``level`` is interpreted as
        the TARGET TOTAL scorable errors across the repo (not per-category).
        This produces the 50/100/200/300 points on the vertical figure.
        """
        import time

        from bioguider.managers.config import TOTAL_ERROR_LEVELS

        if total_levels is None:
            total_levels = TOTAL_ERROR_LEVELS

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        benchmark_dir = os.path.join(output_base_path, f"gradient_{timestamp}")
        os.makedirs(benchmark_dir, exist_ok=True)

        self.print_step(
            "GradientStart", f"Testing total-error levels: {total_levels}"
        )

        file_selection = self.select_target_files(
            baseline_repo_path, max_per_category=self.config.max_files_per_category
        )

        results: Dict[int, StressTestResult] = {}

        for total in total_levels:
            start_time = time.time()
            self.print_step(
                f"GradientLevel:{total}", f"Starting with target_total={total}"
            )

            level_dir = os.path.join(benchmark_dir, f"total_{total}")
            tmp_repo_path = os.path.join(level_dir, "tmp_repo")

            self.prepare_tmp_repo(baseline_repo_path, tmp_repo_path)
            project_terms = self.extract_project_terms(tmp_repo_path)

            # Inject with a TOTAL budget (min_per_category is derived inside).
            injection_results = self.inject_errors_parallel(
                file_selection,
                tmp_repo_path,
                min_per_category=1,  # floor; real value derived from target_total_errors
                project_terms=project_terms,
                target_total_errors=total,
            )

            self.save_injection_manifest(
                injection_results, level_dir, "BENCHMARK_MANIFEST.json"
            )

            injected_files = (
                list(injection_results.keys())
                if self.config.limit_generation_files
                else None
            )
            max_files = len(injected_files) if injected_files else None

            gen = DocumentationGenerationManager(self.llm, self.step_callback)
            out_dir = gen.run(
                report_path=report_path,
                repo_path=tmp_repo_path,
                target_files=injected_files,
                max_files=max_files,
            )

            self.print_step("EvaluateFixes", "Computing benchmark metrics...")
            manifests = self.convert_injections_to_manifests(injection_results)
            evaluator = UnifiedMetricsEvaluator(
                llm=self.llm if self.config.detect_semantic_fp else None,
                detect_fp=self.config.detect_semantic_fp,
            )
            eval_result = evaluator.evaluate_multiple_files(manifests, out_dir)

            duration = time.time() - start_time

            results[total] = StressTestResult(
                error_count=total,
                evaluation_result=eval_result,
                output_dir=level_dir,
                duration_seconds=duration,
            )

            self._save_level_results(results[total], level_dir)

            # Use the scorable F1 as the headline when available (D2 field).
            f1 = getattr(eval_result, "f1_score_scorable", eval_result.f1_score)
            self.print_step(
                f"GradientLevelComplete:{total}",
                f"F1_scorable={f1:.3f}, FixRate={eval_result.fix_rate:.3f}",
            )

        self._save_stress_test_results(results, benchmark_dir)
        self.print_step("GradientComplete", f"Results saved to {benchmark_dir}")
        return results

    # =========================================================================
    # Required Abstract Methods
    # =========================================================================

    def run_quant_test(
        self,
        report_path: str,
        baseline_repo_path: str,
        tmp_repo_path: str,
        min_per_category: int = 3,
    ) -> str:
        """
        Run a single quantifiable test (delegates to stress test with one level).
        """
        results = self.run_stress_test(
            report_path=report_path,
            baseline_repo_path=baseline_repo_path,
            output_base_path=os.path.dirname(tmp_repo_path),
            stress_levels=[min_per_category],
        )

        if results:
            return list(results.values())[0].output_dir
        return tmp_repo_path

    def run_quant_suite(
        self,
        report_path: str,
        baseline_repo_path: str,
        base_tmp_repo_path: str,
        levels: Dict[str, int],
    ) -> Dict[str, str]:
        """
        Run test suite across multiple named levels.
        """
        stress_levels = list(levels.values())
        results = self.run_stress_test(
            report_path=report_path,
            baseline_repo_path=baseline_repo_path,
            output_base_path=base_tmp_repo_path,
            stress_levels=stress_levels,
        )

        # Map level names to output dirs
        return {
            name: results[count].output_dir
            for name, count in levels.items()
            if count in results
        }

    # =========================================================================
    # Multi-Model Comparison
    # =========================================================================

    def prepare_model_comparison(
        self,
        report_path: str,
        baseline_repo_path: str,
        output_base_path: str,
        error_count: int = 20,
    ) -> str:
        """
        Prepare corrupted files for multi-model comparison.

        Generates corrupted files that can be manually processed by
        different models (GPT-4, Claude, Gemini, etc.) for comparison.

        Args:
            report_path: Path to evaluation report
            baseline_repo_path: Path to baseline repository
            output_base_path: Base output path
            error_count: Number of errors to inject per category

        Returns:
            Path to the prepared benchmark directory
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        benchmark_dir = os.path.join(output_base_path, f"model_comparison_{timestamp}")
        os.makedirs(benchmark_dir, exist_ok=True)

        self.print_step("PrepareComparison", "Preparing files for model comparison")

        # Select files
        file_selection = self.select_target_files(
            baseline_repo_path, max_per_category=self.config.max_files_per_category
        )

        # Create corrupted repo
        corrupted_dir = os.path.join(benchmark_dir, "corrupted")
        self.prepare_tmp_repo(baseline_repo_path, corrupted_dir)

        # Extract terms and inject
        project_terms = self.extract_project_terms(corrupted_dir)
        injection_results = self.inject_errors_parallel(
            file_selection, corrupted_dir, error_count, project_terms
        )

        # Save manifest
        self.save_injection_manifest(
            injection_results, benchmark_dir, "BENCHMARK_MANIFEST.json"
        )

        # Save original files for reference
        originals_dir = os.path.join(benchmark_dir, "originals")
        os.makedirs(originals_dir, exist_ok=True)
        for rel_path, result in injection_results.items():
            orig_save_path = os.path.join(originals_dir, rel_path)
            os.makedirs(os.path.dirname(orig_save_path), exist_ok=True)
            write_file(orig_save_path, result.baseline_content)

        # Create directories for each model
        for model in self.config.comparison_models:
            os.makedirs(os.path.join(benchmark_dir, f"fixed_{model}"), exist_ok=True)

        # Generate instructions
        self._generate_comparison_instructions(benchmark_dir, injection_results)

        self.print_step("ComparisonPrepared", f"Files ready in {benchmark_dir}")
        return benchmark_dir

    def evaluate_model_comparison(
        self,
        benchmark_dir: str,
        models: Optional[List[str]] = None,
    ) -> ModelComparisonResult:
        """
        Evaluate and compare results from multiple models.

        Args:
            benchmark_dir: Path to benchmark directory with fixed files
            models: List of model names to evaluate

        Returns:
            ModelComparisonResult with comparison data
        """
        if models is None:
            models = self.config.comparison_models

        # Load manifest
        manifest_path = os.path.join(benchmark_dir, "BENCHMARK_MANIFEST.json")
        manifest_data = self.load_injection_manifest(manifest_path)

        # Reconstruct manifests dict
        originals_dir = os.path.join(benchmark_dir, "originals")
        corrupted_dir = os.path.join(benchmark_dir, "corrupted")

        all_manifests = {}
        for rel_path, file_info in manifest_data.get("files", {}).items():
            orig_content = read_file(os.path.join(originals_dir, rel_path)) or ""
            corr_content = read_file(os.path.join(corrupted_dir, rel_path)) or ""

            all_manifests[rel_path] = {
                "category": file_info.get("category", ""),
                "manifest": {"errors": file_info.get("errors", [])},
                "baseline_content": orig_content,
                "corrupted_content": corr_content,
            }

        total_errors = manifest_data.get("total_errors", 0)

        result = ModelComparisonResult(
            models=models,
            error_count=total_errors,
        )

        evaluator = UnifiedMetricsEvaluator(
            llm=self.llm if self.config.detect_semantic_fp else None,
            detect_fp=self.config.detect_semantic_fp,
        )

        for model in models:
            model_fixed_dir = os.path.join(benchmark_dir, f"fixed_{model}")

            if not os.path.exists(model_fixed_dir):
                self.print_step(f"SkipModel:{model}", "No fixed files found")
                continue

            # Check if directory has files
            has_files = any(
                os.path.isfile(os.path.join(model_fixed_dir, f))
                for f in os.listdir(model_fixed_dir)
            )
            if not has_files:
                self.print_step(f"SkipModel:{model}", "Directory empty")
                continue

            self.print_step(f"EvaluateModel:{model}", "Computing metrics...")

            eval_result = evaluator.evaluate_multiple_files(
                all_manifests, model_fixed_dir
            )
            result.results[model] = eval_result

            self.print_step(
                f"ModelEvaluated:{model}",
                f"F1={eval_result.f1_score:.3f}, FixRate={eval_result.fix_rate:.3f}",
            )

        # Save comparison results
        self._save_comparison_results(result, benchmark_dir)

        return result

    # =========================================================================
    # Result Export
    # =========================================================================

    def _save_level_results(self, result: StressTestResult, output_dir: str):
        """Save results for a single stress level."""
        results_path = os.path.join(output_dir, "BENCHMARK_RESULTS.json")
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "error_count": result.error_count,
                    "duration_seconds": result.duration_seconds,
                    **result.evaluation_result.to_dict(),
                },
                f,
                indent=2,
            )

    def _save_stress_test_results(
        self,
        results: Dict[int, StressTestResult],
        output_dir: str,
    ):
        """Save aggregate stress test results as JSON and CSV."""
        # JSON format
        stress_results = []
        for level, result in sorted(results.items()):
            stress_results.append(
                {
                    "error_count": level,
                    "duration_seconds": result.duration_seconds,
                    **result.evaluation_result.to_dict(),
                }
            )

        json_path = os.path.join(output_dir, "STRESS_TEST_RESULTS.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump({"stress_results": stress_results}, f, indent=2)

        # CSV format
        csv_path = os.path.join(output_dir, "STRESS_TEST_TABLE.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "error_count",
                    "true_positives",
                    "false_negatives",
                    "false_positives",
                    "precision",
                    "recall",
                    "f1_score",
                    "fix_rate",
                    "duration_seconds",
                ]
            )
            for level, result in sorted(results.items()):
                er = result.evaluation_result
                writer.writerow(
                    [
                        level,
                        er.true_positives,
                        er.false_negatives,
                        er.false_positives,
                        round(er.precision, 4),
                        round(er.recall, 4),
                        round(er.f1_score, 4),
                        round(er.fix_rate, 4),
                        round(result.duration_seconds, 2),
                    ]
                )

        # Markdown report
        self._generate_stress_test_report(results, output_dir)

    def _generate_stress_test_report(
        self,
        results: Dict[int, StressTestResult],
        output_dir: str,
    ):
        """Generate markdown report for stress test."""
        lines = [
            "# Stress Test Results\n",
            f"\n**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n",
            "\n---\n",
            "\n## Summary Table\n",
            "\n| Errors | TP | FN | FP | Precision | Recall | F1 | Fix Rate |\n",
            "|--------|----|----|-----|-----------|--------|-----|----------|\n",
        ]

        for level, result in sorted(results.items()):
            er = result.evaluation_result
            lines.append(
                f"| {level} | {er.true_positives} | {er.false_negatives} | "
                f"{er.false_positives} | {er.precision:.3f} | {er.recall:.3f} | "
                f"{er.f1_score:.3f} | {er.fix_rate:.3f} |\n"
            )

        lines.append("\n---\n")
        lines.append("\n## Key Findings\n")

        # Find performance drop-off point
        prev_f1 = 1.0
        drop_point = None
        for level, result in sorted(results.items()):
            if result.evaluation_result.f1_score < prev_f1 * 0.8:
                drop_point = level
                break
            prev_f1 = result.evaluation_result.f1_score

        if drop_point:
            lines.append(
                f"\n- **Performance drop-off**: Significant decline at {drop_point} errors\n"
            )
        else:
            lines.append("\n- **Performance**: Stable across all tested error levels\n")

        # Best/worst performance
        best_level = max(
            results.keys(), key=lambda k: results[k].evaluation_result.f1_score
        )
        worst_level = min(
            results.keys(), key=lambda k: results[k].evaluation_result.f1_score
        )

        lines.append(
            f"- **Best F1**: {results[best_level].evaluation_result.f1_score:.3f} at {best_level} errors\n"
        )
        lines.append(
            f"- **Worst F1**: {results[worst_level].evaluation_result.f1_score:.3f} at {worst_level} errors\n"
        )

        report_path = os.path.join(output_dir, "STRESS_TEST_REPORT.md")
        with open(report_path, "w", encoding="utf-8") as f:
            f.writelines(lines)

    def _save_comparison_results(
        self,
        result: ModelComparisonResult,
        output_dir: str,
    ):
        """Save model comparison results as JSON and CSV."""
        # JSON format
        comparison_data = {
            "models": result.models,
            "error_count": result.error_count,
            "results": {model: er.to_dict() for model, er in result.results.items()},
        }

        json_path = os.path.join(output_dir, "MODEL_COMPARISON_RESULTS.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(comparison_data, f, indent=2)

        # CSV format
        csv_path = os.path.join(output_dir, "MODEL_COMPARISON_TABLE.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "model",
                    "true_positives",
                    "false_negatives",
                    "false_positives",
                    "precision",
                    "recall",
                    "f1_score",
                    "fix_rate",
                ]
            )
            for model, er in result.results.items():
                writer.writerow(
                    [
                        model,
                        er.true_positives,
                        er.false_negatives,
                        er.false_positives,
                        round(er.precision, 4),
                        round(er.recall, 4),
                        round(er.f1_score, 4),
                        round(er.fix_rate, 4),
                    ]
                )

        # Markdown report
        self._generate_comparison_report(result, output_dir)

    def _generate_comparison_report(
        self,
        result: ModelComparisonResult,
        output_dir: str,
    ):
        """Generate markdown report for model comparison."""
        lines = [
            "# Model Comparison Results\n",
            f"\n**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n",
            f"**Error Count**: {result.error_count}\n",
            "\n---\n",
            "\n## Comparison Table\n",
            "\n| Model | TP | FN | FP | Precision | Recall | F1 | Fix Rate |\n",
            "|-------|----|----|-----|-----------|--------|-----|----------|\n",
        ]

        for model, er in result.results.items():
            lines.append(
                f"| {model} | {er.true_positives} | {er.false_negatives} | "
                f"{er.false_positives} | {er.precision:.3f} | {er.recall:.3f} | "
                f"{er.f1_score:.3f} | {er.fix_rate:.3f} |\n"
            )

        lines.append("\n---\n")
        lines.append("\n## Rankings\n")

        # Rank by F1 score
        ranked = sorted(
            result.results.items(), key=lambda x: x[1].f1_score, reverse=True
        )
        lines.append("\n### By F1 Score\n")
        for i, (model, er) in enumerate(ranked, 1):
            lines.append(f"{i}. **{model}**: {er.f1_score:.3f}\n")

        report_path = os.path.join(output_dir, "MODEL_COMPARISON_REPORT.md")
        with open(report_path, "w", encoding="utf-8") as f:
            f.writelines(lines)

    def _generate_comparison_instructions(
        self,
        output_dir: str,
        injection_results: Dict[str, InjectionResult],
    ):
        """Generate instructions for running model comparison."""
        files_list = list(injection_results.keys())

        lines = [
            "# Model Comparison Instructions\n",
            f"\n**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n",
            "\n---\n",
            "\n## Overview\n",
            f"\nThis benchmark contains {len(files_list)} corrupted files for testing.\n",
            "\n## Files to Process\n",
        ]

        for rel_path in files_list:
            lines.append(f"- `corrupted/{rel_path}`\n")

        lines.append("\n---\n")
        lines.append("\n## Instructions for Each Model\n")

        for model in self.config.comparison_models:
            if model == "bioguider":
                lines.append(f"\n### {model}\n")
                lines.append("Run automatically via the benchmark evaluation.\n")
            else:
                lines.append(f"\n### {model}\n")
                lines.append("1. Open each file in `corrupted/` with your IDE\n")
                lines.append(f"2. Use {model} as the AI model\n")
                lines.append(
                    "3. Prompt: 'Fix all errors, typos, broken links, and formatting issues'\n"
                )
                lines.append(
                    f"4. Save fixed files to `fixed_{model}/` maintaining directory structure\n"
                )

        lines.append("\n---\n")
        lines.append("\n## After Fixing\n")
        lines.append("\nRun evaluation:\n")
        lines.append("```python\n")
        lines.append(
            "from bioguider.managers.benchmark_manager import BenchmarkManager\n"
        )
        lines.append("mgr = BenchmarkManager(llm, callback)\n")
        lines.append(f'result = mgr.evaluate_model_comparison("{output_dir}")\n')
        lines.append("```\n")

        instructions_path = os.path.join(output_dir, "INSTRUCTIONS.md")
        with open(instructions_path, "w", encoding="utf-8") as f:
            f.writelines(lines)
