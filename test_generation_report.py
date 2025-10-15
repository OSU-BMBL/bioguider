#!/usr/bin/env python3

import os
import tempfile
from pathlib import Path

def test_generation_report():
    """Test generation report creation with current fixes"""
    try:
        from bioguider.generation.report_loader import EvaluationReportLoader
        from bioguider.generation.suggestion_extractor import SuggestionExtractor
        from bioguider.generation.change_planner import ChangePlanner
        from bioguider.generation.repo_reader import RepoReader
        from bioguider.generation.style_analyzer import StyleAnalyzer
        from bioguider.generation.models import StyleProfile
        
        print("=== TESTING GENERATION REPORT CREATION ===")
        
        # Step 1: Load and extract
        loader = EvaluationReportLoader()
        report, _ = loader.load('logs/evaluation_report_github_satijalab_seurat.json')
        
        extractor = SuggestionExtractor()
        suggestions = extractor.extract(report)
        
        # Step 2: Read files and plan
        reader = RepoReader("outputs/satijalab_seurat")
        files, missing = reader.read_default_targets()
        
        style_analyzer = StyleAnalyzer()
        style = style_analyzer.analyze(files)
        
        planner = ChangePlanner()
        plan = planner.build_plan("outputs/satijalab_seurat", style, suggestions, files)
        
        # Step 3: Create test generation report
        print("\nCreating test generation report...")
        
        # Focus on specific tutorial files
        test_files = ['vignettes/dim_reduction_vignette.Rmd', 'vignettes/get_started.Rmd']
        
        for file_name in test_files:
            print(f"\n=== {file_name} ===")
            
            # Get suggestions for this file
            file_suggestions = [s for s in suggestions if file_name in s.target_files]
            print(f"Suggestions: {len(file_suggestions)}")
            
            # Get planned edits for this file
            file_edits = [e for e in plan.planned_edits if e.file_path == file_name]
            print(f"Planned edits: {len(file_edits)}")
            
            # Show specific suggestions
            for i, suggestion in enumerate(file_suggestions[:5]):  # Show first 5
                print(f"  {i+1}. {suggestion.action}")
                print(f"     Guidance: {suggestion.content_guidance[:100]}...")
                print(f"     Evidence: {suggestion.source.get('evidence', 'N/A')[:100]}...")
            
            # Show planned edits
            for i, edit in enumerate(file_edits[:5]):  # Show first 5
                print(f"  Edit {i+1}: {edit.edit_type}")
                print(f"    Anchor: {edit.anchor}")
                print(f"    Rationale: {edit.rationale[:100]}...")
        
        # Step 4: Test generation report format
        print("\n=== TESTING GENERATION REPORT FORMAT ===")
        
        # Simulate the generation report writing
        lines = []
        lines.append("# Documentation Improvements Report")
        lines.append("")
        lines.append("**Repository:** outputs/satijalab_seurat")
        lines.append("**Generated:** test_output")
        lines.append("")
        lines.append("## Overview")
        lines.append("")
        lines.append("This report summarizes documentation improvements made based on evaluation feedback.")
        lines.append("")
        
        # Group by file
        files_with_changes = {}
        for edit in plan.planned_edits:
            file_path = edit.file_path
            if file_path not in files_with_changes:
                files_with_changes[file_path] = []
            files_with_changes[file_path].append(edit)
        
        for file_path, edits in files_with_changes.items():
            if file_path in test_files:  # Only show test files
                lines.append(f"### {file_path}")
                lines.append(f"**Changes made:** {len(edits)} improvement(s)")
                lines.append("")
                
                for edit in edits:
                    # Find corresponding suggestion
                    suggestion = next((s for s in suggestions if s.id == edit.suggestion_id), None)
                    
                    if suggestion:
                        guidance = suggestion.content_guidance
                        evidence = suggestion.source.get('evidence', '')
                        
                        # Convert technical action names to user-friendly descriptions
                        action_desc = {
                            'append_section': f'Added "{edit.anchor.get("value", "section")}" section',
                            'replace_intro_block': f'Improved "{edit.anchor.get("value", "section")}" section',
                            'improve_readability': f'Improved readability in "{edit.anchor.get("value", "section")}"',
                            'improve_setup': f'Enhanced setup instructions in "{edit.anchor.get("value", "section")}"',
                            'improve_reproducibility': f'Improved reproducibility in "{edit.anchor.get("value", "section")}"',
                            'improve_structure': f'Enhanced structure in "{edit.anchor.get("value", "section")}"',
                            'improve_code_quality': f'Improved code quality in "{edit.anchor.get("value", "section")}"',
                            'improve_verification': f'Enhanced result verification in "{edit.anchor.get("value", "section")}"',
                            'improve_performance': f'Added performance notes in "{edit.anchor.get("value", "section")}"',
                        }.get(edit.edit_type, f'Improved {edit.edit_type}')
                        
                        lines.append(f"- **{action_desc}**")
                        
                        # Show evaluation score that triggered this improvement
                        score = suggestion.source.get("score", "") if suggestion and suggestion.source else ""
                        if score:
                            lines.append(f"  - *Reason:* Evaluation score was '{score}' - needs improvement")
                        
                        if guidance:
                            lines.append(f"  - *What was updated:* {guidance}")
                        
                        lines.append("")
        
        # Write test report
        test_report_path = "test_generation_report.md"
        with open(test_report_path, 'w') as f:
            f.write('\n'.join(lines))
        
        print(f"Test generation report written to: {test_report_path}")
        
        # Show summary
        print(f"\n=== SUMMARY ===")
        print(f"Total suggestions: {len(suggestions)}")
        print(f"Total planned edits: {len(plan.planned_edits)}")
        print(f"Files with changes: {len(files_with_changes)}")
        print(f"Test files covered: {len([f for f in test_files if f in files_with_changes])}")
        
        return True
        
    except Exception as e:
        print(f"Error in test generation report: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_generation_report()
    if success:
        print("\n✓ Generation report test completed successfully!")
    else:
        print("\n✗ Generation report test failed!")
