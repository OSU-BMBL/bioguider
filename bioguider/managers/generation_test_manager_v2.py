from __future__ import annotations

import os
import json
import shutil
from typing import Dict, List, Tuple
from pathlib import Path

from bioguider.generation.llm_injector import LLMErrorInjector
from bioguider.managers.generation_manager import DocumentationGenerationManager
from bioguider.agents.agent_utils import read_file, write_file


class GenerationTestManagerV2:
    """
    Enhanced version that:
    1. Injects errors into ALL files in multiple categories (README, ALL tutorials, ALL userguides, ALL installation docs)
    2. Tracks errors comprehensively across all files with detailed per-file and per-category statistics
    3. Provides detailed statistics on injected, detected, and fixed errors
    4. Simplifies reporting (fixed vs unchanged, no confusing dual-status)
    5. Saves corrupted, original, and fixed versions for each file for full audit trail
    """
    
    def __init__(self, llm, step_callback):
        self.llm = llm
        self.step_output = step_callback

    def print_step(self, name: str, out: str | None = None):
        if self.step_output:
            self.step_output(step_name=name, step_output=out)

    def _select_target_files(self, baseline_repo_path: str) -> Dict[str, List[str]]:
        """
        Select target files for error injection across multiple categories.
        
        Returns:
            Dict mapping category names to file paths
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
        
        # Tutorial files (RMarkdown vignettes) - ALL FILES
        vignettes_dir = os.path.join(baseline_repo_path, "vignettes")
        if os.path.isdir(vignettes_dir):
            tutorial_files = []
            for f in os.listdir(vignettes_dir):
                if f.endswith('.Rmd') and not f.startswith('.'):
                    tutorial_files.append(os.path.join(vignettes_dir, f))
            # Inject into ALL tutorial files
            targets["tutorial"] = sorted(tutorial_files)
        
        # Installation files
        install_files = []
        for pattern in ["install", "INSTALL", "installation"]:
            for ext in [".md", ".Rmd", ".rst"]:
                fpath = os.path.join(baseline_repo_path, pattern + ext)
                if os.path.exists(fpath):
                    install_files.append(fpath)
        
        # Also check vignettes for installation guides
        if os.path.isdir(vignettes_dir):
            for f in os.listdir(vignettes_dir):
                if "install" in f.lower() and (f.endswith('.Rmd') or f.endswith('.md')):
                    fpath = os.path.join(vignettes_dir, f)
                    if fpath not in install_files:  # Avoid duplicates
                        install_files.append(fpath)
        
        targets["installation"] = install_files  # ALL installation docs
        
        # Userguide files - ALL FILES
        docs_dir = os.path.join(baseline_repo_path, "docs")
        if os.path.isdir(docs_dir):
            userguide_files = []
            for f in os.listdir(docs_dir):
                if f.endswith('.md') and not f.startswith('.'):
                    userguide_files.append(os.path.join(docs_dir, f))
            targets["userguide"] = userguide_files  # ALL userguide files
        
        return targets

    def _inject_errors_into_files(
        self, 
        target_files: Dict[str, List[str]], 
        tmp_repo_path: str, 
        min_per_category: int
    ) -> Dict[str, Dict]:
        """
        Inject errors into selected files.
        
        Returns:
            Dict mapping file paths to injection manifests
        """
        injector = LLMErrorInjector(self.llm)
        all_manifests = {}
        
        for category, file_list in target_files.items():
            self.print_step(f"InjectErrors:{category.title()}", f"Injecting {min_per_category} errors per file into {len(file_list)} files")
            
            for fpath in file_list:
                if not os.path.exists(fpath):
                    continue
                
                baseline_content = read_file(fpath) or ""
                if not baseline_content.strip():
                    continue
                
                try:
                    # Inject errors
                    corrupted, manifest = injector.inject(
                        baseline_content, 
                        min_per_category=min_per_category
                    )
                    
                    # Save corrupted version to tmp repo
                    rel_path = os.path.relpath(fpath, os.path.dirname(os.path.dirname(fpath)))
                    if rel_path.startswith("../"):
                        rel_path = os.path.basename(fpath)
                    
                    corrupted_path = os.path.join(tmp_repo_path, rel_path)
                    os.makedirs(os.path.dirname(corrupted_path), exist_ok=True)
                    write_file(corrupted_path, corrupted)
                    
                    # Track manifest - add file_path to each error for tracking
                    errors_with_file = []
                    for error in manifest.get("errors", []):
                        error_with_file = error.copy()
                        error_with_file["file_path"] = rel_path
                        errors_with_file.append(error_with_file)
                    
                    manifest_with_file = manifest.copy()
                    manifest_with_file["errors"] = errors_with_file
                    
                    all_manifests[rel_path] = {
                        "category": category,
                        "original_path": fpath,
                        "corrupted_path": corrupted_path,
                        "manifest": manifest_with_file,
                        "baseline_content": baseline_content,
                        "corrupted_content": corrupted
                    }
                    
                    self.print_step(
                        f"Injected:{os.path.basename(fpath)}", 
                        f"{len(manifest.get('errors', []))} errors"
                    )
                except Exception as e:
                    self.print_step(f"InjectionError:{os.path.basename(fpath)}", str(e))
                    continue
        
        return all_manifests

    def _evaluate_all_fixes(
        self, 
        all_manifests: Dict[str, Dict], 
        output_dir: str
    ) -> Dict:
        """
        Evaluate fixes across all injected files.
        
        Returns comprehensive statistics.
        """
        from bioguider.generation.test_metrics import evaluate_fixes
        
        all_results = {
            "per_file": {},
            "aggregate": {
                "total_files_injected": len(all_manifests),
                "total_errors_injected": 0,
                "total_errors_fixed": 0,
                "total_errors_unchanged": 0,
                "by_category": {},
                "by_file_type": {}
            },
            "detailed_errors": []
        }
        
        for rel_path, info in all_manifests.items():
            # Read the fixed version
            fixed_path = os.path.join(output_dir, rel_path)
            if not os.path.exists(fixed_path):
                # File wasn't processed - copy original
                fixed_content = info["baseline_content"]
            else:
                fixed_content = read_file(fixed_path) or info["baseline_content"]
            
            # Evaluate fixes for this file
            results = evaluate_fixes(
                info["baseline_content"],
                info["corrupted_content"],
                fixed_content,
                info["manifest"]
            )
            
            # Store per-file results
            all_results["per_file"][rel_path] = {
                "category": info["category"],
                "results": results
            }
            
            # Aggregate statistics
            totals = results.get("summary", {}).get("totals", {})
            file_total_errors = totals.get("total_errors", 0)
            file_fixed = totals.get("fixed_to_baseline", 0) + totals.get("fixed_to_valid", 0)
            file_unchanged = totals.get("unchanged", 0)
            
            all_results["aggregate"]["total_errors_injected"] += file_total_errors
            all_results["aggregate"]["total_errors_fixed"] += file_fixed
            all_results["aggregate"]["total_errors_unchanged"] += file_unchanged
            
            # By file type
            file_cat = info["category"]
            if file_cat not in all_results["aggregate"]["by_file_type"]:
                all_results["aggregate"]["by_file_type"][file_cat] = {
                    "files": 0,
                    "errors_injected": 0,
                    "errors_fixed": 0,
                    "errors_unchanged": 0
                }
            
            all_results["aggregate"]["by_file_type"][file_cat]["files"] += 1
            all_results["aggregate"]["by_file_type"][file_cat]["errors_injected"] += file_total_errors
            all_results["aggregate"]["by_file_type"][file_cat]["errors_fixed"] += file_fixed
            all_results["aggregate"]["by_file_type"][file_cat]["errors_unchanged"] += file_unchanged
            
            # By error category
            for err_cat, metrics in results.get("per_category", {}).items():
                if err_cat not in all_results["aggregate"]["by_category"]:
                    all_results["aggregate"]["by_category"][err_cat] = {
                        "total": 0,
                        "fixed": 0,
                        "unchanged": 0
                    }
                
                all_results["aggregate"]["by_category"][err_cat]["total"] += metrics.get("total", 0)
                all_results["aggregate"]["by_category"][err_cat]["fixed"] += (
                    metrics.get("fixed_to_baseline", 0) + metrics.get("fixed_to_valid", 0)
                )
                all_results["aggregate"]["by_category"][err_cat]["unchanged"] += metrics.get("unchanged", 0)
            
            # Collect detailed errors
            for err in results.get("per_error", []):
                err_detail = {
                    "file": rel_path,
                    "file_category": file_cat,
                    **err
                }
                # Simplify status
                if err["status"] in ("fixed_to_baseline", "fixed_to_valid"):
                    err_detail["status"] = "fixed"
                all_results["detailed_errors"].append(err_detail)
        
        # Calculate aggregate success rate
        total_errors = all_results["aggregate"]["total_errors_injected"]
        fixed_errors = all_results["aggregate"]["total_errors_fixed"]
        all_results["aggregate"]["success_rate"] = (
            round((fixed_errors / total_errors * 100.0), 2) if total_errors > 0 else 0.0
        )
        
        return all_results

    def _generate_comprehensive_report(
        self, 
        results: Dict, 
        output_dir: str,
        level: str
    ):
        """Generate a comprehensive markdown report"""
        agg = results["aggregate"]
        
        lines = [
            "# 🔬 BioGuider Quantifiable Testing Results\n",
            f"**Test Level**: {level.upper()}\n",
            "\n---\n",
            "\n## 📊 Executive Summary\n",
            f"\n### Overall Performance\n",
            f"- **Success Rate**: {agg['success_rate']}%\n",
            f"- **Total Files Tested**: {agg['total_files_injected']}\n",
            f"- **Total Errors Injected**: {agg['total_errors_injected']}\n",
            f"- **Errors Fixed**: {agg['total_errors_fixed']} ({round(agg['total_errors_fixed']/agg['total_errors_injected']*100, 1) if agg['total_errors_injected'] > 0 else 0}%)\n",
            f"- **Errors Unchanged**: {agg['total_errors_unchanged']} ({round(agg['total_errors_unchanged']/agg['total_errors_injected']*100, 1) if agg['total_errors_injected'] > 0 else 0}%)\n",
            "\n---\n",
            "\n## 📂 Performance by File Type\n",
        ]
        
        for file_type, metrics in sorted(agg["by_file_type"].items()):
            fix_rate = (metrics["errors_fixed"] / metrics["errors_injected"] * 100) if metrics["errors_injected"] > 0 else 0
            lines.append(f"\n### {file_type.title()}\n")
            lines.append(f"- Files Tested: {metrics['files']}\n")
            lines.append(f"- Errors Injected: {metrics['errors_injected']}\n")
            lines.append(f"- Errors Fixed: {metrics['errors_fixed']} ({fix_rate:.1f}%)\n")
            lines.append(f"- Errors Unchanged: {metrics['errors_unchanged']}\n")
        
        lines.append("\n---\n")
        lines.append("\n## 🏷️ Performance by Error Category\n")
        lines.append("\n| Category | Total | Fixed | Unchanged | Fix Rate |\n")
        lines.append("|----------|-------|-------|-----------|----------|\n")
        
        for err_cat, metrics in sorted(agg["by_category"].items(), key=lambda x: -x[1]["total"]):
            fix_rate = (metrics["fixed"] / metrics["total"] * 100) if metrics["total"] > 0 else 0
            lines.append(
                f"| {err_cat} | {metrics['total']} | {metrics['fixed']} | "
                f"{metrics['unchanged']} | {fix_rate:.1f}% |\n"
            )
        
        lines.append("\n---\n")
        lines.append("\n## 📝 Detailed Error Breakdown\n")
        
        # Group by file
        by_file = {}
        for err in results["detailed_errors"]:
            fpath = err["file"]
            if fpath not in by_file:
                by_file[fpath] = []
            by_file[fpath].append(err)
        
        for fpath, errors in sorted(by_file.items()):
            fixed_count = sum(1 for e in errors if e["status"] == "fixed")
            total_count = len(errors)
            lines.append(f"\n### `{fpath}`\n")
            lines.append(f"- **Total Errors**: {total_count}\n")
            lines.append(f"- **Fixed**: {fixed_count}\n")
            lines.append(f"- **Unchanged**: {total_count - fixed_count}\n")
            lines.append("\n| ID | Category | Status |\n")
            lines.append("|--------|----------|--------|\n")
            for err in errors:
                lines.append(f"| {err['id']} | {err['category']} | {err['status']} |\n")
        
        lines.append("\n---\n")
        lines.append("\n## 💡 Notes\n")
        lines.append("- Original, corrupted, and fixed versions saved for each file\n")
        lines.append("- Detailed injection manifests available in `INJECTION_MANIFEST.json`\n")
        lines.append("- Complete results data in `GEN_TEST_RESULTS.json`\n")
        
        with open(os.path.join(output_dir, "GEN_TEST_REPORT.md"), "w", encoding="utf-8") as f:
            f.write("".join(lines))

    def run_quant_test(
        self, 
        report_path: str, 
        baseline_repo_path: str, 
        tmp_repo_path: str, 
        min_per_category: int = 3
    ) -> str:
        """
        Run quantifiable testing with multi-file error injection.
        """
        # 1. Select target files across categories
        self.print_step("SelectFiles", "Identifying target files...")
        target_files = self._select_target_files(baseline_repo_path)
        
        total_targets = sum(len(files) for files in target_files.values())
        self.print_step("TargetsSelected", f"{total_targets} files selected across {len(target_files)} categories")
        
        # 2. Copy baseline to tmp (for unmodified files)
        if os.path.exists(tmp_repo_path):
            shutil.rmtree(tmp_repo_path)
        shutil.copytree(baseline_repo_path, tmp_repo_path, symlinks=False, ignore=shutil.ignore_patterns('.git'))
        
        # 3. Inject errors into selected files
        self.print_step("InjectErrors", f"Injecting {min_per_category} errors per category...")
        all_manifests = self._inject_errors_into_files(target_files, tmp_repo_path, min_per_category)
        
        total_errors = sum(len(info["manifest"].get("errors", [])) for info in all_manifests.values())
        self.print_step("InjectionComplete", f"{total_errors} errors injected across {len(all_manifests)} files")
        
        # Save combined injection manifest with proper structure
        # Flatten all errors with file information for easy tracking
        all_errors_flat = []
        files_info = {}
        for rel_path, info in all_manifests.items():
            file_errors = info["manifest"].get("errors", [])
            files_info[rel_path] = {
                "category": info["category"],
                "original_path": info["original_path"],
                "corrupted_path": info["corrupted_path"],
                "error_count": len(file_errors),
                "errors": file_errors
            }
            all_errors_flat.extend(file_errors)
        
        combined_manifest = {
            "total_files": len(all_manifests),
            "total_errors": total_errors,
            "files": files_info,
            "errors": all_errors_flat  # Flat list for easy evaluation
        }
        inj_path = os.path.join(tmp_repo_path, "INJECTION_MANIFEST.json")
        with open(inj_path, "w", encoding="utf-8") as f:
            json.dump(combined_manifest, f, indent=2)
        
        # 4. Run generation/fixing
        self.print_step("RunGeneration", "Running BioGuider to fix errors...")
        gen = DocumentationGenerationManager(self.llm, self.step_output)
        out_dir = gen.run(report_path=report_path, repo_path=tmp_repo_path)
        
        # 5. Evaluate fixes
        self.print_step("EvaluateFixes", "Evaluating error corrections...")
        results = self._evaluate_all_fixes(all_manifests, out_dir)
        
        # 6. Save results
        with open(os.path.join(out_dir, "GEN_TEST_RESULTS.json"), "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        
        # Copy injection manifest to output
        shutil.copy(inj_path, os.path.join(out_dir, "INJECTION_MANIFEST.json"))
        
        # 7. Generate report
        level = "custom"
        if min_per_category <= 3:
            level = "low"
        elif min_per_category <= 7:
            level = "mid"
        else:
            level = "high"
        
        self._generate_comprehensive_report(results, out_dir, level)
        
        # 8. Save versioned baseline files (original and corrupted versions)
        for rel_path, info in all_manifests.items():
            base_name = os.path.basename(rel_path)
            base_dir = os.path.dirname(rel_path)
            
            # Extract file extension properly
            if '.' in base_name:
                name_parts = base_name.rsplit('.', 1)
                base_name_no_ext = name_parts[0]
                ext = '.' + name_parts[1]
            else:
                base_name_no_ext = base_name
                ext = ''
            
            # Create original and corrupted filenames
            orig_name = f"{base_name_no_ext}.original{ext}"
            corr_name = f"{base_name_no_ext}.corrupted{ext}"
            
            # Determine save directory - preserve directory structure
            if base_name == "README.md":
                # Special handling for README - save at root level
                save_dir = out_dir
            else:
                # Save in same directory structure as original
                save_dir = os.path.join(out_dir, base_dir) if base_dir else out_dir
            
            os.makedirs(save_dir, exist_ok=True)
            
            # Save original and corrupted versions
            write_file(os.path.join(save_dir, orig_name), info["baseline_content"])
            write_file(os.path.join(save_dir, corr_name), info["corrupted_content"])
        
        self.print_step("TestComplete", f"Results saved to {out_dir}")
        return out_dir

    def run_quant_suite(
        self, 
        report_path: str, 
        baseline_repo_path: str, 
        base_tmp_repo_path: str, 
        levels: dict[str, int]
    ) -> dict:
        """
        Run test suite across multiple levels.
        """
        results = {}
        for level, min_cnt in levels.items():
            self.print_step(f"RunLevel:{level.upper()}", f"Running with {min_cnt} errors per file")
            tmp_repo_path = f"{base_tmp_repo_path}_{level}"
            out_dir = self.run_quant_test(report_path, baseline_repo_path, tmp_repo_path, min_per_category=min_cnt)
            results[level] = out_dir
        return results

