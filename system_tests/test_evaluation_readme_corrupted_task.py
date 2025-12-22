import pytest
import json
from pathlib import Path
from datetime import datetime

from bioguider.agents.evaluation_readme_task import EvaluationREADMETask


def generate_evaluation_report(
    corrupted_dir: Path,
    injected_errors: list,
    evaluations: dict,
    readme_file: str = "README.corrupted.md"
):
    """
    Generate a comprehensive markdown report comparing injected errors with evaluation results.
    """
    report_lines = []
    
    # Header
    report_lines.append("# Error Detection Evaluation Report\n")
    report_lines.append(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    report_lines.append(f"**Test File**: {readme_file}\n")
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
    
    # Section 2: Evaluation Results
    report_lines.append("\n## 🔍 Evaluation Results\n")
    
    if readme_file in evaluations:
        eval_result = evaluations[readme_file]
        
        report_lines.append(f"\n### Overall Assessment\n")
        report_lines.append(f"- **Project Level**: {eval_result.project_level}\n")
        
        if eval_result.structured_evaluation:
            struct_eval = eval_result.structured_evaluation
            report_lines.append(f"- **Available Score**: {struct_eval.available_score}\n")
            report_lines.append(f"- **Readability Score**: {struct_eval.readability_score}/100\n")
            report_lines.append(f"- **Overall Score**: {struct_eval.overall_score}/100\n")
            
            # Error Detection Summary (most important for this test)
            report_lines.append(f"\n### 🎯 Error Detection Summary\n")
            total_detected = struct_eval.readability_error_count if hasattr(struct_eval, 'readability_error_count') and struct_eval.readability_error_count else 0
            detection_rate = (total_detected / len(injected_errors) * 100) if injected_errors else 0
            
            report_lines.append(f"- **Errors Injected**: {len(injected_errors)}\n")
            report_lines.append(f"- **Issues Detected**: {total_detected}\n")
            report_lines.append(f"- **Detection Rate**: {detection_rate:.1f}%\n")
            
            if detection_rate < 30:
                report_lines.append(f"- **Status**: ❌ **LOW** - Most errors were missed\n")
            elif detection_rate < 60:
                report_lines.append(f"- **Status**: ⚠️ **MODERATE** - Many errors still missed\n")
            elif detection_rate < 85:
                report_lines.append(f"- **Status**: ✓ **GOOD** - Majority of errors detected\n")
            elif detection_rate >= 100:
                report_lines.append(f"- **Status**: ✅ **EXCELLENT** - All errors detected (plus additional issues found)\n")
            else:
                report_lines.append(f"- **Status**: ✅ **EXCELLENT** - Most errors detected\n")
            
            # Show difference analysis
            if total_detected > len(injected_errors):
                report_lines.append(f"- **Note**: LLM detected {total_detected - len(injected_errors)} additional issues beyond injected errors\n")
            elif total_detected < len(injected_errors):
                report_lines.append(f"- **Note**: ⚠️ LLM missed {len(injected_errors) - total_detected} injected errors\n")
            
            # Detected errors list
            if hasattr(struct_eval, 'readability_errors_found') and struct_eval.readability_errors_found:
                report_lines.append(f"\n### 🔍 Detected Issues & Errors\n")
                report_lines.append(f"**Count**: {len(struct_eval.readability_errors_found)} issues found\n\n")
                for i, err in enumerate(struct_eval.readability_errors_found, 1):
                    report_lines.append(f"{i}. {err}\n")
            else:
                report_lines.append(f"\n### Detected Issues\n")
                report_lines.append("⚠️ No error details available in structured format\n")
            
            # Other suggestions (non-error related)
            if struct_eval.readability_suggestions:
                report_lines.append(f"\n### 💡 General Improvement Recommendations\n")
                report_lines.append(f"{struct_eval.readability_suggestions}\n")
    else:
        report_lines.append(f"\n⚠️ **No evaluation results found for {readme_file}**\n")
    
    report_lines.append("\n---\n")
    
    # Section 3: Detection Analysis by Category
    report_lines.append("\n## 📊 Detection Analysis by Error Type\n")
    
    if readme_file in evaluations and evaluations[readme_file].structured_evaluation:
        struct_eval = evaluations[readme_file].structured_evaluation
        
        # Analyze what was detected
        suggestions_text = ""
        if struct_eval.readability_suggestions:
            suggestions_text = struct_eval.readability_suggestions.lower()
        
        if hasattr(struct_eval, 'readability_errors_found') and struct_eval.readability_errors_found:
            for err in struct_eval.readability_errors_found:
                suggestions_text += " " + err.lower()
        
        # Map injected categories to detected mentions
        category_analysis = {}
        for cat, count in error_categories.items():
            detected = False
            evidence = []
            
            # Check for category-specific keywords
            if cat == "typo":
                if "typo" in suggestions_text or "spelling" in suggestions_text or "misspell" in suggestions_text:
                    detected = True
                    evidence.append("typo/spelling corrections found")
            elif cat == "link":
                if "link" in suggestions_text or "url" in suggestions_text or "http" in suggestions_text or "://" in suggestions_text:
                    detected = True
                    evidence.append("link/URL issues found")
            elif cat == "markdown" or cat == "markdown_syntax":
                if "markdown" in suggestions_text or "#" in suggestions_text or "header" in suggestions_text:
                    detected = True
                    evidence.append("markdown issues found")
            elif cat == "bio_term":
                if "genomics" in suggestions_text or "single cell" in suggestions_text or "genomis" in suggestions_text or "sell" in suggestions_text or "bio" in suggestions_text:
                    detected = True
                    evidence.append("bio term errors found")
            elif cat == "image_syntax":
                if "image" in suggestions_text or "![" in suggestions_text:
                    detected = True
                    evidence.append("image syntax issues found")
            
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
        total_detected = struct_eval.readability_error_count if hasattr(struct_eval, 'readability_error_count') and struct_eval.readability_error_count else 0
        individual_rate = (total_detected / len(injected_errors) * 100) if injected_errors else 0
        
        report_lines.append(f"- **Individual Error Detection**: {total_detected}/{len(injected_errors)} errors = {individual_rate:.1f}%\n")
        report_lines.append(f"- **Category Coverage**: {detected_count}/{len(category_analysis)} types = {cat_detection_rate:.1f}%\n")
        
        if individual_rate >= 100 and cat_detection_rate >= 80:
            report_lines.append(f"- **Quality**: ✅ **EXCELLENT** - All errors detected plus additional issues found\n")
        elif individual_rate >= 70 and cat_detection_rate >= 80:
            report_lines.append(f"- **Quality**: ✅ **EXCELLENT** - High detection across all categories\n")
        elif individual_rate >= 50 and cat_detection_rate >= 60:
            report_lines.append(f"- **Quality**: ✓ **GOOD** - Decent coverage, but room for improvement\n")
        elif individual_rate >= 30 or cat_detection_rate >= 40:
            report_lines.append(f"- **Quality**: ⚠️ **FAIR** - Significant gaps in detection\n")
        else:
            report_lines.append(f"- **Quality**: ❌ **POOR** - Major improvements needed\n")
    else:
        report_lines.append("\n⚠️ **Cannot perform detection analysis - no evaluation data available**\n")
    
    report_lines.append("\n---\n")
    
    # Section 4: Recommendations
    report_lines.append("\n## 💡 Recommendations\n")
    
    if readme_file in evaluations and evaluations[readme_file].structured_evaluation:
        struct_eval = evaluations[readme_file].structured_evaluation
        detected_count = struct_eval.readability_error_count if hasattr(struct_eval, 'readability_error_count') else 0
        injected_count = len(injected_errors)
        detection_rate = (detected_count / injected_count * 100) if injected_count else 0
        
        # Find which categories were missed
        missed_categories = []
        for cat, info in category_analysis.items():
            if not info["detected"]:
                missed_categories.append(f"{cat} ({info['injected']} errors)")
        
        if detected_count > injected_count:
            report_lines.append(f"\n### ✅ Excellent Detection - Over-Detection is Good\n")
            report_lines.append(f"- Detected {detected_count}/{injected_count} = {detection_rate:.1f}%\n")
            report_lines.append(f"- Found {detected_count - injected_count} additional issues beyond injected errors\n")
            report_lines.append(f"- This shows the LLM is thorough in finding real issues\n")
            report_lines.append(f"- Extra errors may be:\n")
            report_lines.append(f"  * Related issues (e.g., same typo appearing multiple times)\n")
            report_lines.append(f"  * Cascade errors (one error causing perception of multiple)\n")
            report_lines.append(f"  * Genuine additional issues the LLM found\n")
            report_lines.append(f"- Review the detailed list to identify false positives vs real additional issues\n\n")
        elif detected_count < injected_count:
            report_lines.append(f"⚠️ **Under-detection** ({injected_count - detected_count} errors missed):\n")
            report_lines.append(f"- Some injected errors were not detected\n")
            report_lines.append(f"- Review missed error categories below\n\n")
        else:
            report_lines.append(f"✅ **Perfect match** - All injected errors were detected!\n\n")
    
    # Section 5: Recommendations
    report_lines.append("\n## 💡 Recommendations\n")
    
    if readme_file in evaluations and evaluations[readme_file].structured_evaluation:
        struct_eval = evaluations[readme_file].structured_evaluation
        detected_count = struct_eval.readability_error_count if hasattr(struct_eval, 'readability_error_count') else 0
        
        if detected_count < len(injected_errors) * 0.5:
            report_lines.append("\n### Prompt Improvement Needed\n")
            report_lines.append("- The evaluation detected fewer than 50% of injected errors\n")
            report_lines.append("- Consider adding more specific examples to the prompts\n")
            report_lines.append("- May need to increase token limits further\n")
            report_lines.append("- Consider breaking evaluation into multiple focused passes\n")
        elif detected_count < len(injected_errors) * 0.8:
            report_lines.append("\n### Good Detection, Room for Improvement\n")
            report_lines.append("- The evaluation is detecting most errors\n")
            report_lines.append("- Fine-tune prompts to catch remaining error types\n")
            report_lines.append("- Review missed error categories for pattern\n")
        elif detected_count <= len(injected_errors) * 1.2:
            report_lines.append("\n### Excellent Detection\n")
            report_lines.append("- The evaluation is successfully detecting the injected errors\n")
            report_lines.append("- Some over-detection is normal and shows thoroughness\n")
            report_lines.append("- Continue monitoring with different test cases\n")
        else:
            report_lines.append("\n### High Over-Detection\n")
            report_lines.append(f"- The evaluation detected {detected_count - len(injected_errors)} more errors than injected\n")
            report_lines.append("- This could indicate:\n")
            report_lines.append("  * Very thorough scanning (good)\n")
            report_lines.append("  * Some false positives (review the detailed list)\n")
            report_lines.append("  * Counting related errors separately\n")
            report_lines.append("- Review the complete error list to validate all detections\n")
    
    report_lines.append("\n---\n")
    report_lines.append(f"\n*Report generated by test_evaluation_readme_corrupted_task.py*\n")
    
    # Save report
    report_path = corrupted_dir / "ERROR_DETECTION_EVALUATION_REPORT.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.writelines(report_lines)
    
    print(f"\n✅ Report saved to: {report_path}")
    return report_path

def test_EvaluationReadmeCorruptedTask_Seurat_Low(llm, step_callback):
    """
    Test evaluation of corrupted README with injected errors.
    This test uses a README file with deliberately injected errors to verify
    that the evaluation can detect typos, link issues, markdown problems, and bio term errors.
    """
    # Path to the corrupted README test data
    # Path to the corrupted README test data
    base_dir = Path("outputs/_tmp_satijalab_seurat_low")
    # Find latest timestamp dir
    dirs = sorted([d for d in base_dir.iterdir() if d.is_dir() and d.name[0].isdigit()])
    if not dirs:
        pytest.skip("No output directory found")
    corrupted_dir = dirs[-1]
    print(f"Using output directory: {corrupted_dir}")
    
    corrupted_readme = corrupted_dir / "README.corrupted.md"
    manifest_path = corrupted_dir / "INJECTION_MANIFEST.json"
    
    # Load the injection manifest to know what errors were injected
    with open(manifest_path, 'r') as f:
        injection_manifest = json.load(f)
    
    # Get errors for README.md
    readme_errors = []
    for file_path, file_info in injection_manifest.get("files", {}).items():
        if "README.md" in file_path:
            readme_errors = file_info.get("errors", [])
            break
    
    print(f"\n=== Injected Errors Count: {len(readme_errors)} ===")
    print("Error Categories:")
    error_categories = {}
    for error in readme_errors:
        cat = error.get("category", "unknown")
        error_categories[cat] = error_categories.get(cat, 0) + 1
    for cat, count in error_categories.items():
        print(f"  - {cat}: {count}")
    
    # Run evaluation on the corrupted README
    # Use the corrupted directory as the repo path
    task = EvaluationREADMETask(
        llm=llm,
        repo_path=str(corrupted_dir),
        gitignore_path=str(corrupted_dir / ".gitignore"),  # May not exist, but that's ok
        step_callback=step_callback,
        collected_files=["README.corrupted.md"]  # Explicitly test the corrupted file
    )
    
    evaluations, token_usage, files = task._evaluate(["README.corrupted.md"])
    
    # Basic assertions
    assert evaluations is not None, "Evaluations should not be None"
    assert len(evaluations) > 0, "Should have at least one evaluation"
    assert "README.corrupted.md" in evaluations, "Should evaluate the corrupted README"
    
    # Get the evaluation result
    eval_result = evaluations["README.corrupted.md"]
    
    # Print the evaluation for manual inspection
    print("\n=== Evaluation Result ===")
    print(f"Project Level: {eval_result.project_level}")
    
    if eval_result.structured_evaluation:
        print(f"\nStructured Evaluation:")
        print(f"  - Available: {eval_result.structured_evaluation.available_score}")
        print(f"  - Readability Score: {eval_result.structured_evaluation.readability_score}")
        print(f"  - Readability Suggestions: {eval_result.structured_evaluation.readability_suggestions[:200]}...")
        print(f"  - Overall Score: {eval_result.structured_evaluation.overall_score}")
    
    if eval_result.free_evaluation:
        print(f"\nFree Evaluation:")
        if hasattr(eval_result.free_evaluation, 'readability'):
            print(f"  - Readability: {eval_result.free_evaluation.readability[:300]}...")
    
    # Assertions on the quality of detection
    # The corrupted README should have a lower score due to injected errors
    if eval_result.structured_evaluation:
        # With 15 errors injected, readability should be impacted
        # We expect the evaluation to detect at least some issues
        print(f"\n=== Detection Analysis ===")
        print(f"Readability suggestions length: {len(eval_result.structured_evaluation.readability_suggestions)}")
        
        # Check if common error types are mentioned in suggestions
        suggestions_text = eval_result.structured_evaluation.readability_suggestions.lower()
        
        detected_issues = []
        if "typo" in suggestions_text or "spelling" in suggestions_text or "misspell" in suggestions_text:
            detected_issues.append("typos")
        if "link" in suggestions_text or "url" in suggestions_text or "http" in suggestions_text:
            detected_issues.append("links")
        if "header" in suggestions_text or "markdown" in suggestions_text or "#" in suggestions_text:
            detected_issues.append("markdown")
        if "genomics" in suggestions_text or "single cell" in suggestions_text or "spatial" in suggestions_text:
            detected_issues.append("bio_terms")
        
        print(f"Detected issue types: {detected_issues}")
        print(f"Expected to detect: typos, links, markdown, bio_terms")
        
        # Warning if detection is poor (but don't fail the test yet)
        if len(detected_issues) < 2:
            print("\n⚠️  WARNING: Evaluation may not be detecting enough error types")
            print("   Consider reviewing and updating the evaluation prompts")
    
    # File should be present
    assert files is not None and len(files) > 0
    
    # Generate comprehensive report
    report_path = generate_evaluation_report(
        corrupted_dir=corrupted_dir,
        injected_errors=readme_errors,
        evaluations=evaluations,
        readme_file="README.corrupted.md"
    )
    
    print(f"\n📄 Comprehensive report generated: {report_path}")
    
    return evaluations, readme_errors


def test_EvaluationReadmeCorruptedTask_Compare_Original(llm, step_callback):
    """
    Compare evaluation scores between original and corrupted README.
    The corrupted version should have lower scores.
    """
    base_dir = Path("outputs/_tmp_satijalab_seurat_low")
    # Find latest timestamp dir
    dirs = sorted([d for d in base_dir.iterdir() if d.is_dir() and d.name[0].isdigit()])
    if not dirs:
        pytest.skip("No output directory found")
    corrupted_dir = dirs[-1]
    
    # Evaluate original README
    task_original = EvaluationREADMETask(
        llm=llm,
        repo_path=str(corrupted_dir),
        gitignore_path=str(corrupted_dir / ".gitignore"),
        step_callback=step_callback,
        collected_files=["README.original.md"]
    )
    eval_original, _, _ = task_original._evaluate(["README.original.md"])
    
    # Evaluate corrupted README
    task_corrupted = EvaluationREADMETask(
        llm=llm,
        repo_path=str(corrupted_dir),
        gitignore_path=str(corrupted_dir / ".gitignore"),
        step_callback=step_callback,
        collected_files=["README.corrupted.md"]
    )
    eval_corrupted, _, _ = task_corrupted._evaluate(["README.corrupted.md"])
    
    # Compare scores
    if "README.original.md" in eval_original and "README.corrupted.md" in eval_corrupted:
        orig_result = eval_original["README.original.md"]
        corr_result = eval_corrupted["README.corrupted.md"]
        
        if orig_result.structured_evaluation and corr_result.structured_evaluation:
            orig_score = orig_result.structured_evaluation.overall_score
            corr_score = corr_result.structured_evaluation.overall_score
            
            print(f"\n=== Score Comparison ===")
            print(f"Original README Score: {orig_score}")
            print(f"Corrupted README Score: {corr_score}")
            print(f"Difference: {orig_score - corr_score}")
            
            # The corrupted version should ideally have a lower score
            # But we won't enforce this as a hard requirement yet since we're tuning prompts
            if corr_score >= orig_score:
                print("\n⚠️  WARNING: Corrupted README has same or higher score than original!")
                print("   This suggests the evaluation is not detecting the injected errors")
            
            # Generate comparison report
            comparison_lines = []
            comparison_lines.append("# Comparison Report: Original vs Corrupted README\n")
            comparison_lines.append(f"\n**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            comparison_lines.append("\n---\n")
            
            comparison_lines.append("\n## Score Comparison\n")
            comparison_lines.append(f"- **Original README Score**: {orig_score}/100\n")
            comparison_lines.append(f"- **Corrupted README Score**: {corr_score}/100\n")
            comparison_lines.append(f"- **Difference**: {orig_score - corr_score}\n")
            
            if corr_score < orig_score:
                comparison_lines.append(f"\n✅ **Result**: Corrupted version scored lower (good - errors detected)\n")
            elif corr_score == orig_score:
                comparison_lines.append(f"\n⚠️ **Result**: Scores are equal (evaluation may not be sensitive enough)\n")
            else:
                comparison_lines.append(f"\n❌ **Result**: Corrupted version scored higher (bad - errors not detected)\n")
            
            comparison_lines.append("\n---\n")
            
            # Save comparison
            comparison_path = corrupted_dir / "ORIGINAL_VS_CORRUPTED_COMPARISON.md"
            with open(comparison_path, 'w', encoding='utf-8') as f:
                f.writelines(comparison_lines)
            
            print(f"\n📄 Comparison report saved to: {comparison_path}")
    
    assert eval_original is not None
    assert eval_corrupted is not None

