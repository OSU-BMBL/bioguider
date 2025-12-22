"""
Benchmark Manager for comprehensive error injection testing.

Provides:
- Stress testing across multiple error count levels (10, 20, 40, 60, 100)
- Multi-process parallel execution for files and stress levels
- Multi-model comparison support (BioGuider + external models)
- CSV/JSON export of results
"""
from __future__ import annotations

import os
import json
import csv
import shutil
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable, Tuple

from langchain_openai.chat_models.base import BaseChatOpenAI

from bioguider.generation.llm_injector import LLMErrorInjector
from bioguider.generation.benchmark_metrics import (
    BenchmarkResult,
    BenchmarkEvaluator,
    evaluate_benchmark,
)
from bioguider.managers.generation_manager import DocumentationGenerationManager
from bioguider.agents.agent_utils import read_file, write_file


# Default stress test levels
DEFAULT_STRESS_LEVELS = [10, 20, 40, 60, 100]

# Supported external models for comparison
SUPPORTED_MODELS = ["bioguider", "gpt-5.1", "claude-sonnet", "gemini"]


@dataclass
class StressTestResult:
    """Result of a single stress test level."""
    error_count: int
    benchmark_result: BenchmarkResult
    output_dir: str
    duration_seconds: float = 0.0


@dataclass
class ModelComparisonResult:
    """Comparison results across multiple models."""
    models: List[str]
    error_count: int
    results: Dict[str, BenchmarkResult] = field(default_factory=dict)


class BenchmarkManager:
    """
    Manages comprehensive benchmark testing for error injection.
    
    Features:
    - Stress testing with configurable error levels
    - Multi-process parallel execution
    - Multi-model comparison support
    - Comprehensive result export (JSON, CSV, Markdown)
    """
    
    def __init__(
        self,
        llm: BaseChatOpenAI,
        step_callback: Optional[Callable] = None,
        max_workers: int = 4
    ):
        self.llm = llm
        self.step_callback = step_callback
        self.max_workers = max_workers
    
    def print_step(self, name: str, output: str = ""):
        """Output step progress."""
        if self.step_callback:
            self.step_callback(step_name=name, step_output=output)
        else:
            print(f"[{name}] {output}")
    
    # =========================================================================
    # FILE SELECTION
    # =========================================================================
    
    def _select_target_files(self, baseline_repo_path: str) -> Dict[str, List[str]]:
        """
        Select target files for error injection across multiple categories.
        """
        targets = {
            "readme": [],
            "tutorial": [],
            "userguide": [],
            "installation": []
        }
        
        # README files
        readme_path = os.path.join(baseline_repo_path, "README.md")
        if os.path.exists(readme_path):
            targets["readme"].append(readme_path)
        
        # Tutorial files (RMarkdown vignettes)
        vignettes_dir = os.path.join(baseline_repo_path, "vignettes")
        if os.path.isdir(vignettes_dir):
            for f in sorted(os.listdir(vignettes_dir)):
                if f.endswith('.Rmd') and not f.startswith('.'):
                    targets["tutorial"].append(os.path.join(vignettes_dir, f))
        
        # Installation files
        for pattern in ["install", "INSTALL", "installation"]:
            for ext in [".md", ".Rmd", ".rst"]:
                fpath = os.path.join(baseline_repo_path, pattern + ext)
                if os.path.exists(fpath):
                    targets["installation"].append(fpath)
        
        # Userguide files
        docs_dir = os.path.join(baseline_repo_path, "docs")
        if os.path.isdir(docs_dir):
            for f in sorted(os.listdir(docs_dir)):
                if f.endswith('.md') and not f.startswith('.'):
                    targets["userguide"].append(os.path.join(docs_dir, f))
        
        return targets
    
    def _extract_project_terms(self, repo_path: str) -> List[str]:
        """Extract function names and key terms from the codebase."""
        import re
        from collections import Counter
        
        terms = Counter()
        
        for root, _, files in os.walk(repo_path):
            if ".git" in root or "__pycache__" in root:
                continue
            
            for file in files:
                fpath = os.path.join(root, file)
                try:
                    content = read_file(fpath)
                    if not content:
                        continue
                    
                    if file.endswith(".py"):
                        funcs = re.findall(r"def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(", content)
                        terms.update(funcs)
                        classes = re.findall(r"class\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*[:\(]", content)
                        terms.update(classes)
                    elif file.endswith(".R"):
                        funcs = re.findall(r"([a-zA-Z_.][a-zA-Z0-9_.]*)\s*<-\s*function", content)
                        terms.update(funcs)
                except Exception:
                    continue
        
        filtered = [t for t, _ in terms.most_common(50) 
                   if len(t) > 4 and t not in {"init", "self", "setup", "test", "main"}]
        return filtered[:20]
    
    # =========================================================================
    # ERROR INJECTION
    # =========================================================================
    
    def _inject_errors_into_file(
        self,
        fpath: str,
        category: str,
        tmp_repo_path: str,
        min_per_category: int,
        project_terms: List[str],
        injector: LLMErrorInjector
    ) -> Optional[Dict[str, Any]]:
        """Inject errors into a single file."""
        if not os.path.exists(fpath):
            return None
        
        baseline_content = read_file(fpath) or ""
        if not baseline_content.strip():
            return None
        
        try:
            corrupted, manifest = injector.inject(
                baseline_content,
                min_per_category=min_per_category,
                project_terms=project_terms
            )
            
            # Determine relative path
            rel_path = os.path.relpath(fpath, os.path.dirname(os.path.dirname(fpath)))
            if rel_path.startswith("../"):
                rel_path = os.path.basename(fpath)
            
            corrupted_path = os.path.join(tmp_repo_path, rel_path)
            os.makedirs(os.path.dirname(corrupted_path), exist_ok=True)
            write_file(corrupted_path, corrupted)
            
            # Add file path to each error
            for error in manifest.get("errors", []):
                error["file_path"] = rel_path
            
            return {
                "rel_path": rel_path,
                "category": category,
                "original_path": fpath,
                "corrupted_path": corrupted_path,
                "manifest": manifest,
                "baseline_content": baseline_content,
                "corrupted_content": corrupted,
            }
        except Exception as e:
            self.print_step(f"InjectionError:{os.path.basename(fpath)}", str(e))
            return None
    
    def _inject_errors_parallel(
        self,
        target_files: Dict[str, List[str]],
        tmp_repo_path: str,
        min_per_category: int
    ) -> Dict[str, Dict]:
        """Inject errors into multiple files in parallel."""
        injector = LLMErrorInjector(self.llm)
        project_terms = self._extract_project_terms(tmp_repo_path)
        self.print_step("ExtractTerms", f"Found {len(project_terms)} project terms")
        
        all_manifests = {}
        
        # Flatten file list with categories
        files_with_cats = []
        for category, file_list in target_files.items():
            for fpath in file_list:
                files_with_cats.append((fpath, category))
        
        self.print_step("InjectErrors", f"Injecting into {len(files_with_cats)} files with {min_per_category} errors/category")
        
        # Use ThreadPoolExecutor since LLM calls are I/O bound
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {}
            for fpath, category in files_with_cats:
                future = executor.submit(
                    self._inject_errors_into_file,
                    fpath, category, tmp_repo_path, min_per_category, project_terms, injector
                )
                futures[future] = (fpath, category)
            
            for future in as_completed(futures):
                fpath, category = futures[future]
                try:
                    result = future.result()
                    if result:
                        all_manifests[result["rel_path"]] = result
                        self.print_step(
                            f"Injected:{os.path.basename(fpath)}",
                            f"{len(result['manifest'].get('errors', []))} errors"
                        )
                except Exception as e:
                    self.print_step(f"InjectionFailed:{os.path.basename(fpath)}", str(e))
        
        return all_manifests
    
    # =========================================================================
    # STRESS TESTING
    # =========================================================================
    
    def run_stress_test(
        self,
        report_path: str,
        baseline_repo_path: str,
        output_base_path: str,
        stress_levels: List[int] = None,
        max_files_per_category: int = 10,
        detect_semantic_fp: bool = True,
        limit_generation_files: bool = True
    ) -> Dict[int, StressTestResult]:
        """
        Run stress tests across multiple error count levels.
        
        Args:
            report_path: Path to evaluation report JSON
            baseline_repo_path: Path to baseline repository
            output_base_path: Base path for output directories
            stress_levels: List of error counts to test (default: [10, 20, 40, 60, 100])
            max_files_per_category: Max files to process per category
            detect_semantic_fp: Whether to run semantic FP detection
            
        Returns:
            Dict mapping error_count to StressTestResult
        """
        import time
        
        if stress_levels is None:
            stress_levels = DEFAULT_STRESS_LEVELS
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        benchmark_dir = os.path.join(output_base_path, f"benchmark_{timestamp}")
        os.makedirs(benchmark_dir, exist_ok=True)
        
        self.print_step("StressTestStart", f"Testing levels: {stress_levels}")
        
        # Select target files once
        target_files = self._select_target_files(baseline_repo_path)
        
        # Limit files per category
        for cat in target_files:
            if len(target_files[cat]) > max_files_per_category:
                target_files[cat] = target_files[cat][:max_files_per_category]
        
        total_files = sum(len(v) for v in target_files.values())
        self.print_step("FilesSelected", f"{total_files} files across {len(target_files)} categories")
        
        results: Dict[int, StressTestResult] = {}
        
        for level in stress_levels:
            start_time = time.time()
            self.print_step(f"StressLevel:{level}", f"Starting with {level} errors per category")
            
            # Create level-specific directory
            level_dir = os.path.join(benchmark_dir, f"level_{level}")
            tmp_repo_path = os.path.join(level_dir, "tmp_repo")
            
            # Copy baseline repo
            if os.path.exists(tmp_repo_path):
                shutil.rmtree(tmp_repo_path)
            shutil.copytree(baseline_repo_path, tmp_repo_path, 
                          symlinks=False, ignore=shutil.ignore_patterns('.git'))
            
            # Inject errors
            all_manifests = self._inject_errors_parallel(target_files, tmp_repo_path, level)
            
            total_errors = sum(len(info["manifest"].get("errors", [])) for info in all_manifests.values())
            self.print_step("InjectionComplete", f"{total_errors} errors in {len(all_manifests)} files")
            
            # Save injection manifest
            self._save_manifest(all_manifests, level_dir)
            
            # Run BioGuider to fix - only process injected files to save time
            injected_files = list(all_manifests.keys()) if limit_generation_files else None
            num_injected = len(injected_files) if injected_files else 0
            
            # Always use max_files as a hard limit when limiting
            max_files_limit = num_injected if (limit_generation_files and num_injected > 0) else None
            
            if limit_generation_files:
                self.print_step("RunGeneration", f"Processing ONLY {num_injected} injected files (max_files={max_files_limit})")
            else:
                self.print_step("RunGeneration", "Processing ALL files...")
            
            gen = DocumentationGenerationManager(self.llm, self.step_callback)
            out_dir = gen.run(
                report_path=report_path, 
                repo_path=tmp_repo_path,
                target_files=injected_files,  # Filter by file path (primary)
                max_files=max_files_limit      # Hard limit (backup guarantee)
            )
            
            # Evaluate results
            self.print_step("EvaluateFixes", "Computing benchmark metrics...")
            benchmark_result = evaluate_benchmark(
                all_manifests, out_dir, self.llm, detect_semantic_fp
            )
            
            duration = time.time() - start_time
            
            results[level] = StressTestResult(
                error_count=level,
                benchmark_result=benchmark_result,
                output_dir=level_dir,
                duration_seconds=duration,
            )
            
            # Save level results
            self._save_level_results(results[level], level_dir)
            
            self.print_step(
                f"LevelComplete:{level}",
                f"F1={benchmark_result.f1_score:.3f}, FixRate={benchmark_result.fix_rate:.3f}"
            )
        
        # Save aggregate stress test results
        self._save_stress_test_results(results, benchmark_dir)
        
        self.print_step("StressTestComplete", f"Results saved to {benchmark_dir}")
        return results
    
    # =========================================================================
    # MULTI-MODEL COMPARISON
    # =========================================================================
    
    def prepare_model_comparison(
        self,
        report_path: str,
        baseline_repo_path: str,
        output_base_path: str,
        error_count: int = 20,
        max_files_per_category: int = 10
    ) -> str:
        """
        Prepare corrupted files for multi-model comparison.
        
        This generates corrupted files that can be manually run through
        Cursor with different models (GPT-5.1, Claude Sonnet, Gemini).
        
        Args:
            report_path: Path to evaluation report
            baseline_repo_path: Path to baseline repository
            output_base_path: Base output path
            error_count: Number of errors to inject per category
            max_files_per_category: Max files per category
            
        Returns:
            Path to the prepared benchmark directory
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        benchmark_dir = os.path.join(output_base_path, f"model_comparison_{timestamp}")
        os.makedirs(benchmark_dir, exist_ok=True)
        
        self.print_step("PrepareComparison", f"Preparing files for model comparison")
        
        # Select and limit files
        target_files = self._select_target_files(baseline_repo_path)
        for cat in target_files:
            if len(target_files[cat]) > max_files_per_category:
                target_files[cat] = target_files[cat][:max_files_per_category]
        
        # Create tmp repo for injection
        tmp_repo_path = os.path.join(benchmark_dir, "corrupted")
        if os.path.exists(tmp_repo_path):
            shutil.rmtree(tmp_repo_path)
        shutil.copytree(baseline_repo_path, tmp_repo_path,
                       symlinks=False, ignore=shutil.ignore_patterns('.git'))
        
        # Inject errors
        all_manifests = self._inject_errors_parallel(target_files, tmp_repo_path, error_count)
        
        # Save manifest
        self._save_manifest(all_manifests, benchmark_dir)
        
        # Save original files for reference
        originals_dir = os.path.join(benchmark_dir, "originals")
        os.makedirs(originals_dir, exist_ok=True)
        for rel_path, info in all_manifests.items():
            orig_save_path = os.path.join(originals_dir, rel_path)
            os.makedirs(os.path.dirname(orig_save_path), exist_ok=True)
            write_file(orig_save_path, info["baseline_content"])
        
        # Create directories for each model's fixed output
        for model in SUPPORTED_MODELS:
            model_dir = os.path.join(benchmark_dir, f"fixed_{model}")
            os.makedirs(model_dir, exist_ok=True)
        
        # Generate instructions file
        self._generate_comparison_instructions(benchmark_dir, all_manifests)
        
        self.print_step("ComparisonPrepared", f"Files ready in {benchmark_dir}")
        return benchmark_dir
    
    def evaluate_model_comparison(
        self,
        benchmark_dir: str,
        models: List[str] = None,
        detect_semantic_fp: bool = True
    ) -> ModelComparisonResult:
        """
        Evaluate and compare results from multiple models.
        
        Args:
            benchmark_dir: Path to benchmark directory with fixed files
            models: List of model names to evaluate
            detect_semantic_fp: Whether to run semantic FP detection
            
        Returns:
            ModelComparisonResult with comparison data
        """
        if models is None:
            models = SUPPORTED_MODELS
        
        # Load manifest
        manifest_path = os.path.join(benchmark_dir, "BENCHMARK_MANIFEST.json")
        with open(manifest_path, 'r') as f:
            manifest_data = json.load(f)
        
        # Reconstruct manifests dict
        all_manifests = {}
        originals_dir = os.path.join(benchmark_dir, "originals")
        corrupted_dir = os.path.join(benchmark_dir, "corrupted")
        
        for rel_path, file_info in manifest_data["files"].items():
            orig_content = read_file(os.path.join(originals_dir, rel_path)) or ""
            corr_content = read_file(os.path.join(corrupted_dir, rel_path)) or ""
            
            all_manifests[rel_path] = {
                "category": file_info["category"],
                "manifest": {"errors": file_info["errors"]},
                "baseline_content": orig_content,
                "corrupted_content": corr_content,
            }
        
        total_errors = manifest_data.get("total_errors", 0)
        
        result = ModelComparisonResult(
            models=models,
            error_count=total_errors,
        )
        
        for model in models:
            model_fixed_dir = os.path.join(benchmark_dir, f"fixed_{model}")
            
            if not os.path.exists(model_fixed_dir):
                self.print_step(f"SkipModel:{model}", "No fixed files found")
                continue
            
            # Check if there are any files in the directory
            has_files = any(os.path.isfile(os.path.join(model_fixed_dir, f)) 
                          for f in os.listdir(model_fixed_dir))
            if not has_files:
                self.print_step(f"SkipModel:{model}", "Directory empty")
                continue
            
            self.print_step(f"EvaluateModel:{model}", "Computing metrics...")
            
            benchmark_result = evaluate_benchmark(
                all_manifests, model_fixed_dir, self.llm, detect_semantic_fp
            )
            
            result.results[model] = benchmark_result
            
            self.print_step(
                f"ModelEvaluated:{model}",
                f"F1={benchmark_result.f1_score:.3f}, FixRate={benchmark_result.fix_rate:.3f}"
            )
        
        # Save comparison results
        self._save_comparison_results(result, benchmark_dir)
        
        return result
    
    # =========================================================================
    # RESULT EXPORT
    # =========================================================================
    
    def _save_manifest(self, all_manifests: Dict[str, Dict], output_dir: str):
        """Save injection manifest to JSON."""
        all_errors = []
        files_info = {}
        
        for rel_path, info in all_manifests.items():
            file_errors = info["manifest"].get("errors", [])
            files_info[rel_path] = {
                "category": info["category"],
                "error_count": len(file_errors),
                "errors": file_errors,
            }
            all_errors.extend(file_errors)
        
        manifest = {
            "total_files": len(all_manifests),
            "total_errors": len(all_errors),
            "files": files_info,
        }
        
        manifest_path = os.path.join(output_dir, "BENCHMARK_MANIFEST.json")
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
    
    def _save_level_results(self, result: StressTestResult, output_dir: str):
        """Save results for a single stress level."""
        results_path = os.path.join(output_dir, "BENCHMARK_RESULTS.json")
        with open(results_path, 'w') as f:
            json.dump({
                "error_count": result.error_count,
                "duration_seconds": result.duration_seconds,
                **result.benchmark_result.to_dict()
            }, f, indent=2)
    
    def _save_stress_test_results(
        self,
        results: Dict[int, StressTestResult],
        output_dir: str
    ):
        """Save aggregate stress test results as JSON and CSV."""
        # JSON format
        stress_results = []
        for level, result in sorted(results.items()):
            stress_results.append({
                "error_count": level,
                "duration_seconds": result.duration_seconds,
                **result.benchmark_result.to_dict()
            })
        
        json_path = os.path.join(output_dir, "STRESS_TEST_RESULTS.json")
        with open(json_path, 'w') as f:
            json.dump({"stress_results": stress_results}, f, indent=2)
        
        # CSV format
        csv_path = os.path.join(output_dir, "STRESS_TEST_TABLE.csv")
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                "error_count", "true_positives", "false_negatives", "false_positives",
                "precision", "recall", "f1_score", "fix_rate", "duration_seconds"
            ])
            for level, result in sorted(results.items()):
                br = result.benchmark_result
                writer.writerow([
                    level, br.true_positives, br.false_negatives, br.false_positives,
                    round(br.precision, 4), round(br.recall, 4), round(br.f1_score, 4),
                    round(br.fix_rate, 4), round(result.duration_seconds, 2)
                ])
        
        # Markdown report
        self._generate_stress_test_report(results, output_dir)
    
    def _generate_stress_test_report(
        self,
        results: Dict[int, StressTestResult],
        output_dir: str
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
            br = result.benchmark_result
            lines.append(
                f"| {level} | {br.true_positives} | {br.false_negatives} | "
                f"{br.false_positives} | {br.precision:.3f} | {br.recall:.3f} | "
                f"{br.f1_score:.3f} | {br.fix_rate:.3f} |\n"
            )
        
        lines.append("\n---\n")
        lines.append("\n## Key Findings\n")
        
        # Find performance drop-off point
        prev_f1 = 1.0
        drop_point = None
        for level, result in sorted(results.items()):
            if result.benchmark_result.f1_score < prev_f1 * 0.8:  # 20% drop
                drop_point = level
                break
            prev_f1 = result.benchmark_result.f1_score
        
        if drop_point:
            lines.append(f"\n- **Performance drop-off**: Significant decline observed at {drop_point} errors\n")
        else:
            lines.append("\n- **Performance**: Stable across all tested error levels\n")
        
        # Best/worst performance
        best_level = max(results.keys(), key=lambda k: results[k].benchmark_result.f1_score)
        worst_level = min(results.keys(), key=lambda k: results[k].benchmark_result.f1_score)
        
        lines.append(f"- **Best F1 Score**: {results[best_level].benchmark_result.f1_score:.3f} at {best_level} errors\n")
        lines.append(f"- **Worst F1 Score**: {results[worst_level].benchmark_result.f1_score:.3f} at {worst_level} errors\n")
        
        report_path = os.path.join(output_dir, "STRESS_TEST_REPORT.md")
        with open(report_path, 'w') as f:
            f.writelines(lines)
    
    def _save_comparison_results(
        self,
        result: ModelComparisonResult,
        output_dir: str
    ):
        """Save model comparison results as JSON and CSV."""
        # JSON format
        comparison_data = {
            "models": result.models,
            "error_count": result.error_count,
            "results": {
                model: br.to_dict()
                for model, br in result.results.items()
            }
        }
        
        json_path = os.path.join(output_dir, "MODEL_COMPARISON_RESULTS.json")
        with open(json_path, 'w') as f:
            json.dump(comparison_data, f, indent=2)
        
        # CSV format
        csv_path = os.path.join(output_dir, "MODEL_COMPARISON_TABLE.csv")
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                "model", "true_positives", "false_negatives", "false_positives",
                "precision", "recall", "f1_score", "fix_rate"
            ])
            for model, br in result.results.items():
                writer.writerow([
                    model, br.true_positives, br.false_negatives, br.false_positives,
                    round(br.precision, 4), round(br.recall, 4),
                    round(br.f1_score, 4), round(br.fix_rate, 4)
                ])
        
        # Markdown report
        self._generate_comparison_report(result, output_dir)
    
    def _generate_comparison_report(
        self,
        result: ModelComparisonResult,
        output_dir: str
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
        
        for model, br in result.results.items():
            lines.append(
                f"| {model} | {br.true_positives} | {br.false_negatives} | "
                f"{br.false_positives} | {br.precision:.3f} | {br.recall:.3f} | "
                f"{br.f1_score:.3f} | {br.fix_rate:.3f} |\n"
            )
        
        lines.append("\n---\n")
        lines.append("\n## Rankings\n")
        
        # Rank by F1 score
        ranked = sorted(result.results.items(), key=lambda x: x[1].f1_score, reverse=True)
        lines.append("\n### By F1 Score\n")
        for i, (model, br) in enumerate(ranked, 1):
            lines.append(f"{i}. **{model}**: {br.f1_score:.3f}\n")
        
        report_path = os.path.join(output_dir, "MODEL_COMPARISON_REPORT.md")
        with open(report_path, 'w') as f:
            f.writelines(lines)
    
    def _generate_comparison_instructions(
        self,
        output_dir: str,
        all_manifests: Dict[str, Dict]
    ):
        """Generate instructions for running model comparison."""
        files_list = list(all_manifests.keys())
        
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
        
        for model in SUPPORTED_MODELS:
            if model == "bioguider":
                lines.append(f"\n### {model}\n")
                lines.append("Run automatically via the benchmark evaluation.\n")
            else:
                lines.append(f"\n### {model}\n")
                lines.append("1. Open each file in `corrupted/` with Cursor\n")
                lines.append(f"2. Use {model} as the AI model\n")
                lines.append("3. Prompt: 'Fix all errors, typos, broken links, and formatting issues in this file'\n")
                lines.append(f"4. Save fixed files to `fixed_{model}/` maintaining directory structure\n")
        
        lines.append("\n---\n")
        lines.append("\n## After Fixing\n")
        lines.append("\nRun evaluation:\n")
        lines.append("```python\n")
        lines.append("from bioguider.managers.benchmark_manager import BenchmarkManager\n")
        lines.append("mgr = BenchmarkManager(llm, callback)\n")
        lines.append(f'result = mgr.evaluate_model_comparison("{output_dir}")\n')
        lines.append("```\n")
        
        instructions_path = os.path.join(output_dir, "INSTRUCTIONS.md")
        with open(instructions_path, 'w') as f:
            f.writelines(lines)

