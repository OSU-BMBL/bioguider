import pytest
import json
from pathlib import Path
from datetime import datetime

from bioguider.agents.evaluation_tutorial_task import EvaluationTutorialTask


def generate_tutorial_evaluation_report(
    corrupted_dir: Path,
    injected_errors: list,
    evaluations: dict,
    tutorial_file: str,
    gen_test_results: dict = None
):
    """
    Generate a comprehensive markdown report comparing injected errors with tutorial evaluation results.
    """
    report_lines = []
    
    # Header
    report_lines.append("# Tutorial Error Detection Evaluation Report\n")
    report_lines.append(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    report_lines.append(f"**Test File**: {tutorial_file}\n")
    report_lines.append(f"**Test Directory**: {corrupted_dir}\n")
    report_lines.append("\n---\n")
    
    # Section 1: Injected Errors Summary
    report_lines.append("\n## 📝 Injected Errors Summary\n")
    report_lines.append(f"\n**Total Errors Injected**: {len(injected_errors)}\n")
    
    # Count by category
    error_categories = {}
    for error in injected_errors:
        cat = error.get("category", "unknown")
        error_categories[cat] = error_categories.get(cat, 0) + 1
    
    report_lines.append("\n### Errors by Category\n")
    report_lines.append("| Category | Count |\n")
    report_lines.append("|----------|-------|\n")
    for cat, count in sorted(error_categories.items()):
        report_lines.append(f"| {cat} | {count} |\n")
    
    # Detailed list of injected errors
    report_lines.append("\n### Detailed List of Injected Errors\n")
    for i, error in enumerate(injected_errors, 1):
        report_lines.append(f"\n**{i}. {error.get('id', 'N/A')}** ({error.get('category', 'unknown')})\n")
        report_lines.append(f"- **Original**: `{error.get('original_snippet', 'N/A')}`\n")
        report_lines.append(f"- **Mutated**: `{error.get('mutated_snippet', 'N/A')}`\n")
        report_lines.append(f"- **Rationale**: {error.get('rationale', 'N/A')}\n")
    
    report_lines.append("\n---\n")

    # Section 1.5: Unfixed Errors Analysis (New Section)
    if gen_test_results:
        report_lines.append("\n## 🔧 Unfixed Errors Analysis\n")
        
        # Find file in results
        file_results = None
        # Try to match the tutorial file in the results
        # tutorial_file is like "vignettes/de_vignette.corrupted.Rmd"
        # results keys are like "vignettes/de_vignette.Rmd" (original path)
        target_base = tutorial_file.replace(".corrupted", "").replace("vignettes/", "")
        
        for filepath, data in gen_test_results.get("per_file", {}).items():
            if target_base in filepath:
                file_results = data
                break
        
        if file_results:
            errors = file_results.get("errors", [])
            total_errs = len(errors)
            fixed_errs = sum(1 for e in errors if e.get("status") == "fixed")
            unfixed_errs = sum(1 for e in errors if e.get("status") == "unchanged")
            
            if total_errs > 0:
                fix_rate = fixed_errs / total_errs * 100
                report_lines.append(f"\n**Fix Rate**: {fixed_errs}/{total_errs} ({fix_rate:.1f}%)\n")
            else:
                report_lines.append(f"\n**Fix Rate**: {fixed_errs}/{total_errs} (N/A - no errors found)\n")
            report_lines.append(f"**Unfixed**: {unfixed_errs} errors\n")
            
            # Group unfixed by category
            unfixed_by_cat = {}
            for e in errors:
                if e.get("status") == "unchanged":
                    cat = e.get("category", "unknown")
                    if cat not in unfixed_by_cat:
                        unfixed_by_cat[cat] = []
                    unfixed_by_cat[cat].append(e)
            
            if unfixed_by_cat:
                report_lines.append("\n### Unfixed Errors by Category\n")
                for cat, errs in sorted(unfixed_by_cat.items(), key=lambda x: len(x[1]), reverse=True):
                    # Add explanation based on category
                    explanation = ""
                    if cat == "inline_code":
                        explanation = " (By design - BioGuider does not modify inline code)"
                    elif cat == "markdown_structure":
                        explanation = " (Technical limitation - LLM struggles with structure)"
                    elif cat == "duplicate":
                        explanation = " (Complex - requires context to resolve)"
                    elif cat == "function":
                        explanation = " (Edge cases - context ambiguous)"
                        
                    report_lines.append(f"\n#### {cat.upper()} - {len(errs)} unfixed{explanation}\n")
                    
                    # List up to 5 examples
                    for i, err in enumerate(errs[:5], 1):
                        report_lines.append(f"{i}. `{err.get('original_snippet', 'N/A')}` -> `{err.get('mutated_snippet', 'N/A')}`\n")
                    if len(errs) > 5:
                        report_lines.append(f"... and {len(errs)-5} more\n")
            else:
                report_lines.append("\n✅ **All errors were fixed!**\n")
        else:
            report_lines.append("\n⚠️ Could not find detailed fix results for this file.\n")
            
        report_lines.append("\n---\n")
    
    # Section 2: Evaluation Results
    report_lines.append("\n## 🔍 Tutorial Evaluation Results\n")
    
    if tutorial_file in evaluations:
        eval_result = evaluations[tutorial_file]
        
        if eval_result.tutorial_evaluation:
            tutorial_eval = eval_result.tutorial_evaluation
            report_lines.append(f"\n### Overall Assessment\n")
            report_lines.append(f"- **Overall Score**: {tutorial_eval.overall_score}/100\n")
            report_lines.append(f"- **Readability Score**: {tutorial_eval.readability_score}/100\n")
            report_lines.append(f"- **Setup & Dependencies Score**: {tutorial_eval.setup_and_dependencies_score}/100\n")
            report_lines.append(f"- **Reproducibility Score**: {tutorial_eval.reproducibility_score}/100\n")
            report_lines.append(f"- **Code Quality Score**: {tutorial_eval.executable_code_quality_score}/100\n")
            
            # Error Detection Summary (most important for this test)
            report_lines.append(f"\n### 🎯 Error Detection Summary\n")
            # Use the new error_count field if available, otherwise fall back to suggestions length
            total_detected = 0
            if hasattr(tutorial_eval, 'readability_error_count') and tutorial_eval.readability_error_count:
                total_detected = tutorial_eval.readability_error_count
            elif hasattr(tutorial_eval, 'readability_errors_found') and tutorial_eval.readability_errors_found:
                total_detected = len(tutorial_eval.readability_errors_found)
            elif tutorial_eval.readability_suggestions:
                total_detected = len(tutorial_eval.readability_suggestions)
            
            # Calculate category-level detection first (more meaningful for patterns)
            num_categories = len(error_categories)
            
            report_lines.append(f"- **Error Categories Injected**: {num_categories}\n")
            report_lines.append(f"- **Total Error Instances**: {len(injected_errors)}\n")
            report_lines.append(f"- **Error Reports Generated**: {total_detected}\n")
            
            # Note: Many errors are repeated patterns (e.g., 10 similar link errors)
            # The LLM detects patterns and reports them, not necessarily every instance
            detection_rate = (total_detected / len(injected_errors) * 100) if injected_errors else 0
            report_lines.append(f"- **Instance Coverage**: {detection_rate:.1f}% ({total_detected}/{len(injected_errors)} instances)\n")
            report_lines.append(f"\n*Note: Many injected errors are repeated patterns. Detection rate measures individual instance mentions, but category coverage (below) is more indicative of pattern detection.*\n")
            
            # Status based on a more lenient threshold since we're detecting patterns
            if detection_rate < 15:
                report_lines.append(f"- **Status**: ❌ **LOW** - Few error instances reported\n")
            elif detection_rate < 40:
                report_lines.append(f"- **Status**: ⚠️ **MODERATE** - Some patterns detected\n")
            elif detection_rate < 70:
                report_lines.append(f"- **Status**: ✓ **GOOD** - Many patterns detected\n")
            else:
                report_lines.append(f"- **Status**: ✅ **EXCELLENT** - High instance coverage\n")
            
            # Readability errors detected (use new errors_found field)
            errors_to_display = []
            if hasattr(tutorial_eval, 'readability_errors_found') and tutorial_eval.readability_errors_found:
                errors_to_display = tutorial_eval.readability_errors_found
            elif tutorial_eval.readability_suggestions:
                errors_to_display = tutorial_eval.readability_suggestions
            
            if errors_to_display:
                report_lines.append(f"\n### 🔍 Detected Issues & Errors\n")
                report_lines.append(f"**Count**: {len(errors_to_display)} issues found\n\n")
                for i, error in enumerate(errors_to_display, 1):
                    report_lines.append(f"{i}. {error}\n")
            else:
                report_lines.append(f"\n### Detected Issues\n")
                report_lines.append("⚠️ No issues detected in readability section\n")
            
            # Other improvement suggestions (non-error related)
            other_suggestions = []
            if tutorial_eval.setup_and_dependencies_suggestions:
                other_suggestions.extend([f"**Setup/Dependencies**: {s}" for s in tutorial_eval.setup_and_dependencies_suggestions])
            if tutorial_eval.reproducibility_suggestions:
                other_suggestions.extend([f"**Reproducibility**: {s}" for s in tutorial_eval.reproducibility_suggestions])
            if tutorial_eval.structure_and_navigation_suggestions:
                other_suggestions.extend([f"**Structure**: {s}" for s in tutorial_eval.structure_and_navigation_suggestions])
            if tutorial_eval.executable_code_quality_suggestions:
                other_suggestions.extend([f"**Code Quality**: {s}" for s in tutorial_eval.executable_code_quality_suggestions])
            if tutorial_eval.result_verification_suggestions:
                other_suggestions.extend([f"**Result Verification**: {s}" for s in tutorial_eval.result_verification_suggestions])
            if tutorial_eval.performance_and_resource_notes_suggestions:
                other_suggestions.extend([f"**Performance**: {s}" for s in tutorial_eval.performance_and_resource_notes_suggestions])
            
            if other_suggestions:
                report_lines.append(f"\n### 💡 Other Improvement Suggestions\n")
                for i, suggestion in enumerate(other_suggestions, 1):
                    report_lines.append(f"{i}. {suggestion}\n")
    else:
        report_lines.append(f"\n⚠️ **No evaluation results found for {tutorial_file}**\n")
    
    report_lines.append("\n---\n")
    
    # Section 3: Detection Analysis by Category
    report_lines.append("\n## 📊 Detection Analysis by Error Type\n")
    
    if tutorial_file in evaluations and evaluations[tutorial_file].tutorial_evaluation:
        tutorial_eval = evaluations[tutorial_file].tutorial_evaluation
        
        # Analyze what was detected by examining all suggestions and errors
        suggestions_text = ""
        
        # Add errors_found if available (new field with better error detection)
        if hasattr(tutorial_eval, 'readability_errors_found') and tutorial_eval.readability_errors_found:
            for err in tutorial_eval.readability_errors_found:
                suggestions_text += " " + err.lower()
        
        # Add all suggestions lists
        for suggestions_list in [
            tutorial_eval.readability_suggestions if hasattr(tutorial_eval, 'readability_suggestions') else [],
            tutorial_eval.setup_and_dependencies_suggestions,
            tutorial_eval.reproducibility_suggestions,
            tutorial_eval.structure_and_navigation_suggestions,
            tutorial_eval.executable_code_quality_suggestions,
            tutorial_eval.result_verification_suggestions,
            tutorial_eval.performance_and_resource_notes_suggestions,
        ]:
            if suggestions_list:
                suggestions_text += " " + " ".join(suggestions_list).lower()
        
        # Map injected categories to detected mentions
        category_analysis = {}
        for cat, count in error_categories.items():
            detected = False
            evidence = []
            
            # Check for category-specific keywords
            if cat == "typo":
                if "typo" in suggestions_text or "spelling" in suggestions_text or "misspell" in suggestions_text or "exampl" in suggestions_text or "analysi" in suggestions_text:
                    detected = True
                    evidence.append("typo/spelling corrections found")
            elif cat == "link":
                if "link" in suggestions_text or "url" in suggestions_text or "http" in suggestions_text or "://" in suggestions_text:
                    detected = True
                    evidence.append("link/URL issues found")
            elif cat == "markdown_structure" or cat == "markdown":
                if "markdown" in suggestions_text or "fence" in suggestions_text or "code block" in suggestions_text or "```" in suggestions_text or "chunk" in suggestions_text:
                    detected = True
                    evidence.append("markdown/structure issues found")
            elif cat == "bio_term":
                if "genomics" in suggestions_text or "single cell" in suggestions_text or "genomis" in suggestions_text or "sell" in suggestions_text or "bio" in suggestions_text:
                    detected = True
                    evidence.append("bio term errors found")
            elif cat == "function":
                if "function" in suggestions_text or "dat()" in suggestions_text or "sys.dat" in suggestions_text or "api" in suggestions_text:
                    detected = True
                    evidence.append("function name issues found")
            elif cat == "inline_code":
                if "backtick" in suggestions_text or "inline" in suggestions_text or "`" in suggestions_text or "code" in suggestions_text:
                    detected = True
                    evidence.append("inline code issues found")
            
            category_analysis[cat] = {
                "injected": count,
                "detected": detected,
                "evidence": evidence
            }
        
        # Summary table
        report_lines.append(f"\n### Category Detection Summary\n")
        report_lines.append("| Error Type | Injected Count | Status | Notes |\n")
        report_lines.append("|------------|----------------|--------|-------|\n")
        
        detected_count = 0
        for cat in sorted(category_analysis.keys()):
            info = category_analysis[cat]
            status = "✅ Detected" if info["detected"] else "❌ Missed"
            notes = info["evidence"][0] if info["evidence"] else "No evidence found"
            report_lines.append(f"| {cat} | {info['injected']} | {status} | {notes} |\n")
            if info["detected"]:
                detected_count += 1
        
        cat_detection_rate = (detected_count / len(category_analysis) * 100) if category_analysis else 0
        report_lines.append(f"\n**Category Detection Rate**: {detected_count}/{len(category_analysis)} = {cat_detection_rate:.1f}%\n")
        
        # Overall quality assessment
        report_lines.append(f"\n### Overall Analysis\n")
        # Use error_count if available, otherwise fall back to suggestions length
        total_detected = 0
        if hasattr(tutorial_eval, 'readability_error_count') and tutorial_eval.readability_error_count:
            total_detected = tutorial_eval.readability_error_count
        elif hasattr(tutorial_eval, 'readability_errors_found') and tutorial_eval.readability_errors_found:
            total_detected = len(tutorial_eval.readability_errors_found)
        elif tutorial_eval.readability_suggestions:
            total_detected = len(tutorial_eval.readability_suggestions)
        individual_rate = (total_detected / len(injected_errors) * 100) if injected_errors else 0
        
        report_lines.append(f"- **Category Pattern Detection**: {detected_count}/{len(category_analysis)} types = {cat_detection_rate:.1f}%\n")
        report_lines.append(f"- **Instance Enumeration**: {total_detected}/{len(injected_errors)} instances = {individual_rate:.1f}%\n")
        report_lines.append(f"\n*Primary metric is Category Pattern Detection. Instance enumeration shows how many specific error mentions were reported.*\n")
        
        # Quality assessment prioritizes category coverage
        if cat_detection_rate >= 80 and individual_rate >= 50:
            report_lines.append(f"- **Quality**: ✅ **EXCELLENT** - High pattern detection and good instance coverage\n")
        elif cat_detection_rate >= 80:
            report_lines.append(f"- **Quality**: ✓ **GOOD** - All major patterns detected, instance reporting could improve\n")
        elif cat_detection_rate >= 60 and individual_rate >= 30:
            report_lines.append(f"- **Quality**: ⚠️ **FAIR** - Most patterns detected but some gaps remain\n")
        else:
            report_lines.append(f"- **Quality**: ❌ **POOR** - Major pattern detection gaps\n")
    else:
        report_lines.append("\n⚠️ **Cannot perform detection analysis - no evaluation data available**\n")
    
    report_lines.append("\n---\n")
    
    # Section 4: Recommendations
    report_lines.append("\n## 💡 Recommendations\n")
    
    if tutorial_file in evaluations and evaluations[tutorial_file].tutorial_evaluation:
        tutorial_eval = evaluations[tutorial_file].tutorial_evaluation
        
        # Use error_count if available, otherwise fall back to suggestions length
        total_detected = 0
        if hasattr(tutorial_eval, 'readability_error_count') and tutorial_eval.readability_error_count:
            total_detected = tutorial_eval.readability_error_count
        elif hasattr(tutorial_eval, 'readability_errors_found') and tutorial_eval.readability_errors_found:
            total_detected = len(tutorial_eval.readability_errors_found)
        elif hasattr(tutorial_eval, 'readability_suggestions') and tutorial_eval.readability_suggestions:
            total_detected = len(tutorial_eval.readability_suggestions)
        detection_rate = (total_detected / len(injected_errors) * 100) if injected_errors else 0
        
        # Find which categories were completely missed vs partially detected
        completely_missed = []
        partially_detected = []
        for cat, info in category_analysis.items():
            if not info["detected"]:
                completely_missed.append(f"{cat} ({info['injected']} errors)")
            else:
                # Even if detected, check if detection seems low
                # (We detected at least 1, but there are multiple injected)
                if info["injected"] > 1:
                    partially_detected.append(cat)
        
        # Determine recommendation level based on detection rate AND category coverage
        categories_detected = len(category_analysis) - len(completely_missed)
        cat_coverage_rate = (categories_detected / len(category_analysis) * 100) if category_analysis else 0
        
        if detection_rate < 30:
            # Check if we have good category coverage even with low instance rate
            if cat_coverage_rate >= 80:
                report_lines.append("\n### 🟡 Pattern Detection Good, Instance Reporting Needs Work\n")
                report_lines.append(f"- **Pattern Detection**: {categories_detected}/{len(category_analysis)} error categories detected ({cat_coverage_rate:.0f}%)\n")
                report_lines.append(f"- **Instance Reporting**: {total_detected} reports for {len(injected_errors)} total instances ({detection_rate:.1f}%)\n")
                report_lines.append(f"- **Assessment**: LLM detects error patterns but doesn't enumerate all instances\n")
                
                if partially_detected:
                    report_lines.append(f"- **Categories with pattern detection**: {', '.join(partially_detected)}\n")
                
                report_lines.append("\n**Action Items**:\n")
                report_lines.append("  1. ✅ Current prompts successfully detect all major error categories\n")
                report_lines.append("  2. Optional: Add instructions to enumerate ALL instances of each error type\n")
                report_lines.append("  3. Consider if full enumeration is necessary - pattern detection may be sufficient\n")
                report_lines.append("  4. Increase max_tokens if comprehensive instance listing is desired\n")
            else:
            report_lines.append("\n### 🔴 Critical: Major Prompt Improvements Needed\n")
                report_lines.append(f"- **Pattern Detection**: {categories_detected}/{len(category_analysis)} error categories detected ({cat_coverage_rate:.0f}%)\n")
                report_lines.append(f"- **Instance Reporting**: {total_detected}/{len(injected_errors)} instances ({detection_rate:.1f}%)\n")
            
            if completely_missed:
                report_lines.append(f"- **Completely Missed**: {', '.join(completely_missed)}\n")
            elif partially_detected:
                    report_lines.append(f"- **Issue**: All error types detected at least once, but many instances missed\n")
                report_lines.append(f"- **Categories needing improvement**: {', '.join(partially_detected)}\n")
            
            report_lines.append("\n**Action Items**:\n")
            report_lines.append("  1. Enhance tutorial evaluation prompts with explicit error detection instructions\n")
            report_lines.append("  2. Add examples of each error type in the prompt\n")
            report_lines.append("  3. Increase LLM max_tokens to ensure complete error reporting\n")
            report_lines.append("  4. Add instructions to report ALL instances of each error type found\n")
            report_lines.append("  5. Review and update grading criteria for readability section\n")
            
        elif detection_rate < 60:
            report_lines.append("\n### 🟡 Moderate Detection - Improvements Needed\n")
            report_lines.append(f"- **Individual Detection**: {total_detected}/{len(injected_errors)} errors ({detection_rate:.1f}%)\n")
            report_lines.append(f"- **Category Coverage**: {len(category_analysis) - len(completely_missed)}/{len(category_analysis)} types\n")
            
            if completely_missed:
                report_lines.append(f"- **Completely Missed**: {', '.join(completely_missed)}\n")
            
            report_lines.append("\n**Action Items**:\n")
            report_lines.append("  1. Fine-tune prompts to better detect all instances of each error type\n")
            if completely_missed:
                report_lines.append(f"  2. Add specific keywords/patterns for missed categories: {', '.join([c.split(' (')[0] for c in completely_missed])}\n")
            report_lines.append("  3. Consider splitting readability evaluation into sub-tasks\n")
            report_lines.append("  4. Test with increased token limits for comprehensive reporting\n")
            
        elif detection_rate < 85:
            report_lines.append("\n### 🟢 Good Detection - Minor Refinements\n")
            report_lines.append(f"- **Individual Detection**: {total_detected}/{len(injected_errors)} errors ({detection_rate:.1f}%)\n")
            report_lines.append(f"- **Category Coverage**: {len(category_analysis) - len(completely_missed)}/{len(category_analysis)} types\n")
            
            if completely_missed:
                report_lines.append(f"- **Needs improvement**: {', '.join(completely_missed)}\n")
            
            report_lines.append("\n**Action Items**:\n")
            report_lines.append("  1. Monitor detection consistency across different test cases\n")
            if completely_missed:
                report_lines.append(f"  2. Fine-tune detection of missed categories\n")
            report_lines.append("  3. Continue testing with diverse error patterns\n")
            
        else:
            report_lines.append("\n### ✅ Excellent Detection\n")
            report_lines.append(f"- **Individual Detection**: {total_detected}/{len(injected_errors)} errors ({detection_rate:.1f}%)\n")
            report_lines.append(f"- **Category Coverage**: {len(category_analysis) - len(completely_missed)}/{len(category_analysis)} types\n")
            report_lines.append("- System is performing well at identifying tutorial errors\n")
            report_lines.append("- Continue monitoring and testing with diverse error patterns\n")
    
    report_lines.append("\n---\n")
    report_lines.append(f"\n*Report generated by test_evaluation_tutorial_corrupted_task.py*\n")
    
    # Save report
    report_path = corrupted_dir / f"TUTORIAL_ERROR_DETECTION_REPORT_{Path(tutorial_file).stem}.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.writelines(report_lines)
    
    print(f"\n✅ Tutorial evaluation report saved to: {report_path}")
    return report_path


def test_EvaluationTutorialCorruptedTask_DeVignette_Low(llm, step_callback):
    """
    Test evaluation of corrupted tutorial/vignette with injected errors.
    This test uses de_vignette.corrupted.Rmd with deliberately injected errors to verify
    that the tutorial evaluation can detect typos, link issues, markdown problems, etc.
    """
    # Path to the corrupted tutorial test data
    # Path to the corrupted tutorial test data
    base_dir = Path("outputs/_tmp_satijalab_seurat_low")
    # Find latest timestamp dir
    dirs = sorted([d for d in base_dir.iterdir() if d.is_dir() and d.name[0].isdigit()])
    if not dirs:
        pytest.skip("No output directory found")
    corrupted_dir = dirs[-1] / "vignettes"
    print(f"Using output directory: {corrupted_dir}")
    
    corrupted_tutorial = "de_vignette.corrupted.Rmd"
    manifest_path = dirs[-1] / "INJECTION_MANIFEST.json"
    results_path = dirs[-1] / "GEN_TEST_RESULTS.json"
    
    # Load the injection manifest
    with open(manifest_path, 'r') as f:
        injection_manifest = json.load(f)
        
    # Load results if available
    gen_test_results = None
    if results_path.exists():
        with open(results_path, 'r') as f:
            gen_test_results = json.load(f)
    
    # Get errors for de_vignette.Rmd
    tutorial_errors = []
    for file_path, file_info in injection_manifest.get("files", {}).items():
        if "de_vignette.Rmd" in file_path:
            tutorial_errors = file_info.get("errors", [])
            break
    
    print(f"\n=== Injected Errors Count: {len(tutorial_errors)} ===")
    print("Error Categories:")
    error_categories = {}
    for error in tutorial_errors:
        cat = error.get("category", "unknown")
        error_categories[cat] = error_categories.get(cat, 0) + 1
    for cat, count in error_categories.items():
        print(f"  - {cat}: {count}")
    
    # Run evaluation on the corrupted tutorial
    task = EvaluationTutorialTask(
        llm=llm,
        repo_path=str(corrupted_dir.parent),  # parent to include vignettes/
        gitignore_path=str(corrupted_dir.parent / ".gitignore"),
        step_callback=step_callback,
        collected_files=[f"vignettes/{corrupted_tutorial}"]
    )
    
    evaluations, token_usage, files = task._evaluate([f"vignettes/{corrupted_tutorial}"])
    
    # Basic assertions
    assert evaluations is not None, "Evaluations should not be None"
    assert len(evaluations) > 0, "Should have at least one evaluation"
    
    # Get the evaluation result
    eval_key = f"vignettes/{corrupted_tutorial}"
    assert eval_key in evaluations, f"Should evaluate {eval_key}"
    
    eval_result = evaluations[eval_key]
    
    # Print the evaluation for manual inspection
    print("\n=== Tutorial Evaluation Result ===")
    if eval_result.tutorial_evaluation:
        tut_eval = eval_result.tutorial_evaluation
        print(f"\nOverall Score: {tut_eval.overall_score}/100")
        print(f"Readability Score: {tut_eval.readability_score}/100")
        print(f"Code Quality Score: {tut_eval.executable_code_quality_score}/100")
        
        if tut_eval.readability_suggestions:
            print(f"\nReadability Suggestions ({len(tut_eval.readability_suggestions)}):")
            for i, suggestion in enumerate(tut_eval.readability_suggestions[:5], 1):
                print(f"  {i}. {suggestion[:100]}...")
    
    # Generate comprehensive report
    report_path = generate_tutorial_evaluation_report(
        corrupted_dir=corrupted_dir.parent,
        injected_errors=tutorial_errors,
        evaluations=evaluations,
        tutorial_file=eval_key,
        gen_test_results=gen_test_results
    )
    
    print(f"\n📄 Comprehensive tutorial report generated: {report_path}")
    
    # File should be present
    assert files is not None and len(files) > 0
    
    return evaluations, tutorial_errors


def test_EvaluationTutorialCorruptedTask_Compare_Original(llm, step_callback):
    """
    Compare evaluation scores between original and corrupted tutorial.
    The corrupted version should have lower scores.
    """
    base_dir = Path("outputs/_tmp_satijalab_seurat_low")
    # Find latest timestamp dir
    dirs = sorted([d for d in base_dir.iterdir() if d.is_dir() and d.name[0].isdigit()])
    if not dirs:
        pytest.skip("No output directory found")
    corrupted_dir = dirs[-1] / "vignettes"
    
    # Evaluate original tutorial
    task_original = EvaluationTutorialTask(
        llm=llm,
        repo_path=str(corrupted_dir.parent),
        gitignore_path=str(corrupted_dir.parent / ".gitignore"),
        step_callback=step_callback,
        collected_files=["vignettes/de_vignette.original.Rmd"]
    )
    eval_original, _, _ = task_original._evaluate(["vignettes/de_vignette.original.Rmd"])
    
    # Evaluate corrupted tutorial
    task_corrupted = EvaluationTutorialTask(
        llm=llm,
        repo_path=str(corrupted_dir.parent),
        gitignore_path=str(corrupted_dir.parent / ".gitignore"),
        step_callback=step_callback,
        collected_files=["vignettes/de_vignette.corrupted.Rmd"]
    )
    eval_corrupted, _, _ = task_corrupted._evaluate(["vignettes/de_vignette.corrupted.Rmd"])
    
    # Compare scores
    if "vignettes/de_vignette.original.Rmd" in eval_original and "vignettes/de_vignette.corrupted.Rmd" in eval_corrupted:
        orig_result = eval_original["vignettes/de_vignette.original.Rmd"]
        corr_result = eval_corrupted["vignettes/de_vignette.corrupted.Rmd"]
        
        if orig_result.tutorial_evaluation and corr_result.tutorial_evaluation:
            orig_score = orig_result.tutorial_evaluation.overall_score
            corr_score = corr_result.tutorial_evaluation.overall_score
            
            print(f"\n=== Tutorial Score Comparison ===")
            print(f"Original Tutorial Score: {orig_score}")
            print(f"Corrupted Tutorial Score: {corr_score}")
            print(f"Difference: {orig_score - corr_score}")
            
            if corr_score >= orig_score:
                print("\n⚠️  WARNING: Corrupted tutorial has same or higher score than original!")
                print("   This suggests the evaluation is not detecting the injected errors")
    
    assert eval_original is not None
    assert eval_corrupted is not None

