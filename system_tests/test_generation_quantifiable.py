import os
import pytest
from bioguider.managers.generation_test_manager_v2 import GenerationTestManagerV2

def test_generation_quantifiable_seurat(llm, step_callback):
    """
    Test error injection and automatic fixing for Seurat documentation.
    
    V2 improvements:
    - Injects errors into multiple file types (README, tutorials, userguides, installation)
    - Increased error counts per level (5, 15, 30)
    - Comprehensive statistics by file type and error category
    - Simplified reporting (fixed vs unchanged)
    - Tracks all injected errors across all files
    """
    report_path = "logs/evaluation_report_github_satijalab_seurat.json"
    baseline_repo_path = "data/.adalflow/repos/satijalab_seurat"
    tmp_repo_path = "outputs/_tmp_satijalab_seurat"

    class FilteredGenerationTestManager(GenerationTestManagerV2):
        def __init__(self, llm, step_callback):
            super().__init__(llm, step_callback)
            self.filtered_files = []
            
        def _select_target_files(self, baseline_repo_path: str):
            targets = super()._select_target_files(baseline_repo_path)
            # Filter to only README and top 10 tutorials
            tutorials = targets.get("tutorial", [])
            # Sort to ensure deterministic selection
            tutorials.sort()
            if len(tutorials) > 10:
                tutorials = tutorials[:10]
            
            new_targets = {
                "readme": targets.get("readme", []),
                "tutorial": tutorials,
                "userguide": [], 
                "installation": []
            }
            
            # Store absolute paths of files we want to keep
            self.filtered_files = []
            for file_list in new_targets.values():
                self.filtered_files.extend(file_list)
                
            return new_targets
        
        def run_quant_test(self, report_path, baseline_repo_path, tmp_repo_path, min_per_category=3):
            # Run parent implementation up to generation step
            import os
            import shutil
            import json
            from bioguider.managers.generation_manager import DocumentationGenerationManager
            from bioguider.agents.agent_utils import write_file
            
            # 1-3: Normal parent implementation (select files, copy repo, inject errors)
            self.print_step("SelectFiles", "Identifying target files...")
            target_files = self._select_target_files(baseline_repo_path)
            
            total_targets = sum(len(files) for files in target_files.values())
            self.print_step("TargetsSelected", f"{total_targets} files selected across {len(target_files)} categories")
            
            if os.path.exists(tmp_repo_path):
                shutil.rmtree(tmp_repo_path)
            shutil.copytree(baseline_repo_path, tmp_repo_path, symlinks=False, ignore=shutil.ignore_patterns('.git'))
            
            self.print_step("InjectErrors", f"Injecting {min_per_category} errors per category...")
            all_manifests = self._inject_errors_into_files(target_files, tmp_repo_path, min_per_category)
            
            total_errors = sum(len(info["manifest"].get("errors", [])) for info in all_manifests.values())
            self.print_step("InjectionComplete", f"{total_errors} errors injected across {len(all_manifests)} files")
            
            # Save injection manifest
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
                "errors": all_errors_flat
            }
            inj_path = os.path.join(tmp_repo_path, "INJECTION_MANIFEST.json")
            with open(inj_path, "w", encoding="utf-8") as f:
                json.dump(combined_manifest, f, indent=2)
            
            # CUSTOM: Delete all non-target files before generation
            self.print_step("CleanupNonTargets", "Removing non-target files from tmp repo...")
            count_deleted = 0
            for root, dirs, files in os.walk(tmp_repo_path, topdown=False):
                if '.git' in root:
                    continue
                
                for file in files:
                    file_path = os.path.join(root, file)
                    rel_path = os.path.relpath(file_path, tmp_repo_path)
                    
                    # Check if this file was in our injection targets
                    is_target = rel_path in all_manifests or file.endswith('.json')
                    
                    if not is_target:
                        try:
                            os.remove(file_path)
                            count_deleted += 1
                        except Exception:
                            pass
                
                # Remove empty directories
                for dir_name in dirs:
                    dir_path = os.path.join(root, dir_name)
                    try:
                        if os.path.isdir(dir_path) and not os.listdir(dir_path):
                            os.rmdir(dir_path)
                    except Exception:
                        pass
            
            self.print_step("CleanupComplete", f"Deleted {count_deleted} non-target files")
            
            # 4. Run generation/fixing - now only 11 files remain in tmp_repo_path
            self.print_step("RunGeneration", "Running BioGuider to fix errors...")
            gen = DocumentationGenerationManager(self.llm, self.step_output)
            out_dir = gen.run(report_path=report_path, repo_path=tmp_repo_path)
            
            # 5-8: Continue with normal parent implementation (evaluate, save results, generate report)
            self.print_step("EvaluateFixes", "Evaluating error corrections...")
            results = self._evaluate_all_fixes(all_manifests, out_dir)
            
            with open(os.path.join(out_dir, "GEN_TEST_RESULTS.json"), "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2)
            
            shutil.copy(inj_path, os.path.join(out_dir, "INJECTION_MANIFEST.json"))
            
            level = "custom"
            if min_per_category <= 3:
                level = "low"
            elif min_per_category <= 7:
                level = "mid"
            else:
                level = "high"
            
            self._generate_comprehensive_report(results, out_dir, level)
            
            # Save versioned baseline files
            for rel_path, info in all_manifests.items():
                base_name = os.path.basename(rel_path)
                base_dir = os.path.dirname(rel_path)
                
                if '.' in base_name:
                    name_parts = base_name.rsplit('.', 1)
                    base_name_no_ext = name_parts[0]
                    ext = '.' + name_parts[1]
                else:
                    base_name_no_ext = base_name
                    ext = ''
                
                orig_name = f"{base_name_no_ext}.original{ext}"
                corr_name = f"{base_name_no_ext}.corrupted{ext}"
                
                if base_name == "README.md":
                    save_dir = out_dir
                else:
                    save_dir = os.path.join(out_dir, base_dir) if base_dir else out_dir
                
                os.makedirs(save_dir, exist_ok=True)
                
                write_file(os.path.join(save_dir, orig_name), info["baseline_content"])
                write_file(os.path.join(save_dir, corr_name), info["corrupted_content"])
            
            self.print_step("TestComplete", f"Results saved to {out_dir}")
            return out_dir
        
        def _inject_errors_into_files(self, target_files, tmp_repo_path, min_per_category):
            # Just call parent implementation, no cleanup here
            return super()._inject_errors_into_files(target_files, tmp_repo_path, min_per_category)

    mgr = FilteredGenerationTestManager(llm, step_callback)
    # Low, Mid, High injection levels with increased error counts
    # Start with just low for testing
    suite = mgr.run_quant_suite(
        report_path,
        baseline_repo_path,
        tmp_repo_path,
        levels={"low": 15},  # Test low first, then add mid and high
    )

    for level, out_dir in suite.items():
        assert os.path.isdir(out_dir), f"Output directory not created for {level}"
        assert os.path.isfile(os.path.join(out_dir, "GEN_TEST_RESULTS.json")), f"Test results JSON missing for {level}"
        assert os.path.isfile(os.path.join(out_dir, "GEN_TEST_REPORT.md")), f"Test report MD missing for {level}"
        assert os.path.isfile(os.path.join(out_dir, "README.original.md")), f"Original README missing for {level}"
        assert os.path.isfile(os.path.join(out_dir, "README.corrupted.md")), f"Corrupted README missing for {level}"
        assert os.path.isfile(os.path.join(out_dir, "README.md")), f"Fixed README missing for {level}"

@pytest.mark.skip(reason="deprecated")
def test_generation_quantifiable_scanpy(llm, step_callback):
    report_path = "logs/scanpy_evaluation_results_20250926.json"
    baseline_repo_path = "data/.adalflow/repos/scverse_scanpy"
    tmp_repo_path = "outputs/_tmp_scverse_scanpy"

    mgr = GenerationTestManager(llm, step_callback)
    # Low, Mid, High injection levels
    suite = mgr.run_quant_suite(
        report_path,
        baseline_repo_path,
        tmp_repo_path,
        levels={"low": 15, "mid": 50, "high": 100},
    )

    for level, out_dir in suite.items():
        assert os.path.isdir(out_dir)
        assert os.path.isfile(os.path.join(out_dir, "GEN_TEST_RESULTS.json"))
        assert os.path.isfile(os.path.join(out_dir, "GEN_TEST_REPORT.md"))
        assert os.path.isfile(os.path.join(out_dir, "README.original.md"))
        assert os.path.isfile(os.path.join(out_dir, "README.corrupted.md"))
        assert os.path.isfile(os.path.join(out_dir, "README.md"))


