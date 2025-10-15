#!/usr/bin/env python3

import os
import sys
from pathlib import Path

def debug_generation_pipeline():
    """Debug the complete generation pipeline"""
    try:
        from bioguider.generation.report_loader import EvaluationReportLoader
        from bioguider.generation.suggestion_extractor import SuggestionExtractor
        from bioguider.generation.change_planner import ChangePlanner
        from bioguider.generation.repo_reader import RepoReader
        from bioguider.generation.style_analyzer import StyleAnalyzer
        from bioguider.generation.models import StyleProfile
        
        print("=== GENERATION PIPELINE DEBUG ===")
        
        # Step 1: Load evaluation report
        print("\n1. Loading evaluation report...")
        loader = EvaluationReportLoader()
        report, _ = loader.load('logs/evaluation_report_github_satijalab_seurat.json')
        print(f"✓ Report loaded: {type(report).__name__}")
        
        # Step 2: Extract suggestions
        print("\n2. Extracting suggestions...")
        extractor = SuggestionExtractor()
        suggestions = extractor.extract(report)
        print(f"✓ Suggestions extracted: {len(suggestions)}")
        
        # Group by category
        by_category = {}
        for s in suggestions:
            category = s.category.split('.')[0] if '.' in s.category else s.category
            by_category[category] = by_category.get(category, 0) + 1
        
        print("Suggestions by category:")
        for category, count in sorted(by_category.items()):
            print(f"  - {category}: {count}")
        
        # Step 3: Read repository files
        print("\n3. Reading repository files...")
        reader = RepoReader("outputs/satijalab_seurat")
        files, missing = reader.read_default_targets()
        print(f"✓ Files read: {len(files)}")
        print(f"✓ Missing files: {len(missing)}")
        
        # Step 4: Analyze style
        print("\n4. Analyzing style...")
        style_analyzer = StyleAnalyzer()
        style = style_analyzer.analyze(files)
        print(f"✓ Style analyzed: {type(style).__name__}")
        
        # Step 5: Plan changes
        print("\n5. Planning changes...")
        planner = ChangePlanner()
        plan = planner.build_plan("outputs/satijalab_seurat", style, suggestions, files)
        print(f"✓ Changes planned: {len(plan.planned_edits)}")
        
        # Analyze planned edits
        by_edit_type = {}
        by_target_file = {}
        for edit in plan.planned_edits:
            edit_type = edit.edit_type
            target_file = edit.file_path
            by_edit_type[edit_type] = by_edit_type.get(edit_type, 0) + 1
            by_target_file[target_file] = by_target_file.get(target_file, 0) + 1
        
        print("Planned edits by type:")
        for edit_type, count in sorted(by_edit_type.items()):
            print(f"  - {edit_type}: {count}")
        
        print("Planned edits by target file:")
        for target_file, count in sorted(by_target_file.items()):
            print(f"  - {target_file}: {count}")
        
        # Step 6: Check suggestion coverage
        print("\n6. Checking suggestion coverage...")
        planned_suggestion_ids = {edit.suggestion_id for edit in plan.planned_edits if edit.suggestion_id}
        all_suggestion_ids = {s.id for s in suggestions}
        
        print(f"Total suggestions: {len(all_suggestion_ids)}")
        print(f"Planned suggestions: {len(planned_suggestion_ids)}")
        print(f"Coverage: {len(planned_suggestion_ids)}/{len(all_suggestion_ids)} ({len(planned_suggestion_ids)/len(all_suggestion_ids)*100:.1f}%)")
        
        # Find unplanned suggestions
        unplanned = all_suggestion_ids - planned_suggestion_ids
        if unplanned:
            print(f"Unplanned suggestions: {len(unplanned)}")
            for suggestion_id in list(unplanned)[:5]:  # Show first 5
                suggestion = next(s for s in suggestions if s.id == suggestion_id)
                print(f"  - {suggestion_id}: {suggestion.category} -> {suggestion.action}")
        
        # Step 7: Check specific tutorial suggestions
        print("\n7. Checking tutorial suggestion coverage...")
        tutorial_suggestions = [s for s in suggestions if s.category.startswith('tutorial')]
        tutorial_planned = [edit for edit in plan.planned_edits if edit.suggestion_id and any(s.id == edit.suggestion_id for s in tutorial_suggestions)]
        
        print(f"Tutorial suggestions: {len(tutorial_suggestions)}")
        print(f"Tutorial planned edits: {len(tutorial_planned)}")
        
        # Check specific tutorial files
        tutorial_files = ['vignettes/dim_reduction_vignette.Rmd', 'vignettes/get_started.Rmd']
        for file_name in tutorial_files:
            file_suggestions = [s for s in tutorial_suggestions if file_name in s.target_files]
            file_planned = [edit for edit in tutorial_planned if edit.file_path == file_name]
            print(f"  {file_name}: {len(file_suggestions)} suggestions -> {len(file_planned)} planned edits")
            
            if file_suggestions:
                print(f"    Suggestion types: {set(s.action for s in file_suggestions)}")
            if file_planned:
                print(f"    Edit types: {set(e.edit_type for e in file_planned)}")
        
        return plan, suggestions
        
    except Exception as e:
        print(f"Error in generation pipeline debug: {e}")
        import traceback
        traceback.print_exc()
        return None, None

if __name__ == "__main__":
    plan, suggestions = debug_generation_pipeline()
