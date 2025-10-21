import os
import pytest

from bioguider.managers.generation_manager import DocumentationGenerationManager


@pytest.mark.skip(reason="Switching to scanpy test")
def test_DocumentationGenerationManager_on_seurat_report(llm, step_callback):
    report_path = "logs/seurat_evaluation_results_20250829_112435.json"
    repo_path = "data/.adalflow/repos/satijalab_seurat"

    mgr = DocumentationGenerationManager(llm, step_callback)
    mgr.prepare_repo(repo_path)

    out_dir = mgr.run(report_path=report_path, repo_path=repo_path)
    assert os.path.isdir(out_dir)

    manifest_path = os.path.join(out_dir, "manifest.json")
    assert os.path.isfile(manifest_path)

    # At least ensures pipeline produced outputs directory; documents may be skipped if source files missing
    # If revised README exists, check it is non-empty
    readme_path = os.path.join(out_dir, "README.md")
    if os.path.isfile(readme_path):
        with open(readme_path, "r", encoding="utf-8") as fobj:
            content = fobj.read()
            assert len(content) > 0


@pytest.mark.skip(reason="deprecated")
def test_DocumentationGenerationManager_on_scanpy_report(llm, step_callback):
    report_path = "logs/scanpy_evaluation_results_20250926.json"
    repo_path = "data/.adalflow/repos/scverse_scanpy"

    mgr = DocumentationGenerationManager(llm, step_callback)
    mgr.prepare_repo(repo_path)

    out_dir = mgr.run(report_path=report_path, repo_path=repo_path)
    assert os.path.isdir(out_dir)

    manifest_path = os.path.join(out_dir, "manifest.json")
    assert os.path.isfile(manifest_path)

    # At least ensures pipeline produced outputs directory; documents may be skipped if source files missing
    # If revised README exists, check it is non-empty
    readme_path = os.path.join(out_dir, "README.md")
    if os.path.isfile(readme_path):
        with open(readme_path, "r", encoding="utf-8") as fobj:
            content = fobj.read()
            assert len(content) > 0

@pytest.mark.skip(reason="deprecated")
def test_DocumentationGenerationManager_on_BioGSP_report(llm, step_callback):
    report_path = "logs/BioGSP_evaluation_results_20251008.json"
    repo_path = "data/.adalflow/repos/BioGSP"

    mgr = DocumentationGenerationManager(llm, step_callback)
    mgr.prepare_repo(repo_path)

    out_dir = mgr.run(report_path=report_path, repo_path=repo_path)
    assert os.path.isdir(out_dir)

    manifest_path = os.path.join(out_dir, "manifest.json")
    assert os.path.isfile(manifest_path)

    # At least ensures pipeline produced outputs directory; documents may be skipped if source files missing
    # If revised README exists, check it is non-empty
    readme_path = os.path.join(out_dir, "README.md")
    if os.path.isfile(readme_path):
        with open(readme_path, "r", encoding="utf-8") as fobj:
            content = fobj.read()
            assert len(content) > 0

@pytest.mark.skip(reason="deprecated")
def test_DocumentationGenerationManager_on_seurat_newformat(llm, step_callback):
    report_path = "logs/evaluation_report_github_satijalab_seurat.json"
    repo_path = "data/.adalflow/repos/satijalab_seurat"

    mgr = DocumentationGenerationManager(llm, step_callback)
    mgr.prepare_repo(repo_path)

    out_dir = mgr.run(report_path=report_path, repo_path=repo_path)
    assert os.path.isdir(out_dir)

    manifest_path = os.path.join(out_dir, "manifest.json")
    assert os.path.isfile(manifest_path)

    # Newflow should produce full replacements and original side-by-side copies
    readme_path = os.path.join(out_dir, "README.md")
    if os.path.isfile(readme_path):
        with open(readme_path, "r", encoding="utf-8") as fobj:
            content = fobj.read()
            assert len(content) > 0
        readme_original = os.path.join(out_dir, "README.original.md")
        assert os.path.isfile(readme_original)

@pytest.mark.skip(reason="deprecated")
def test_DocumentationGenerationManager_on_seurat_newformat_fast(llm, step_callback):
    """Fast test that processes only the first 10 suggestions for quick validation"""
    report_path = "logs/evaluation_report_github_satijalab_seurat.json"
    repo_path = "data/.adalflow/repos/satijalab_seurat"

    # Import the suggestion extractor to limit suggestions
    from bioguider.generation.suggestion_extractor import SuggestionExtractor
    from bioguider.generation.report_loader import EvaluationReportLoader
    from bioguider.generation.change_planner import ChangePlanner
    from bioguider.generation.repo_reader import RepoReader
    from bioguider.generation.style_analyzer import StyleAnalyzer
    import tempfile
    import json
    
    # Load report and extract suggestions
    loader = EvaluationReportLoader()
    report, _ = loader.load(report_path)
    
    extractor = SuggestionExtractor()
    all_suggestions = extractor.extract(report)
    
    # Limit to first 10 suggestions
    limited_suggestions = all_suggestions[:25]
    print(f"Testing with {len(limited_suggestions)} suggestions (out of {len(all_suggestions)} total)")
    
    # Read repo files
    reader = RepoReader(repo_path)
    target_files = set()
    for suggestion in limited_suggestions:
        target_files.update(suggestion.target_files)
    
    files, missing = reader.read_files(list(target_files)) if target_files else reader.read_default_targets()
    
    # Analyze style
    style_analyzer = StyleAnalyzer()
    style = style_analyzer.analyze(files)
    
    # Plan changes with limited suggestions
    planner = ChangePlanner()
    plan = planner.build_plan(repo_path=repo_path, style=style, suggestions=limited_suggestions, available_files=files)
    
    print(f"Planned {len(plan.planned_edits)} edits from {len(limited_suggestions)} suggestions")
    
    # Create a mock generation manager for testing
    class MockGenerationManager:
        def __init__(self, llm, step_callback):
            self.llm = llm
            self.step_callback = step_callback
            self.loader = loader
            self.extractor = extractor
            self.planner = planner
            self.style_analyzer = style_analyzer
            
        def prepare_repo(self, repo_path):
            pass
            
        def run(self, report_path, repo_path):
            # Use our limited suggestions instead of extracting from report
            from bioguider.generation.document_renderer import DocumentRenderer
            from bioguider.generation.llm_content_generator import LLMContentGenerator
            from bioguider.generation.llm_cleaner import LLMCleaner
            from bioguider.generation.output_manager import OutputManager
            import os
            
            # Define original_copy_name function locally
            def original_copy_name(path: str) -> str:
                # Handle all file extensions properly
                if "." in path:
                    base, ext = path.rsplit(".", 1)
                    return f"{base}.original.{ext}"
                return f"{path}.original"
            
            # Initialize components
            renderer = DocumentRenderer()
            llm_gen = LLMContentGenerator(self.llm)
            llm_cleaner = LLMCleaner(self.llm)
            output_manager = OutputManager()
            
            # Use our pre-planned edits
            revised = {}
            diff_stats = {}
            edits_by_file = {}
            for e in plan.planned_edits:
                edits_by_file.setdefault(e.file_path, []).append(e)
            
            # Process each file
            for fpath, edits in edits_by_file.items():
                original_content = files.get(fpath, "")
                content = original_content
                total_stats = {"added_lines": 0}
                
                for e in edits:
                    context = original_content
                    if not e.content_template or e.content_template.strip() == "":
                        suggestion = next((s for s in limited_suggestions if s.id == e.suggestion_id), None) if e.suggestion_id else None
                        if suggestion:
                            if e.edit_type == "full_replace":
                                gen_content, gen_usage = llm_gen.generate_full_document(
                                    target_file=fpath,
                                    evaluation_report={"suggestion": suggestion.content_guidance, "evidence": suggestion.source.get("evidence", "") if suggestion.source else ""},
                                    context=context,
                                )
                                if isinstance(gen_content, str) and gen_content:
                                    e.content_template = gen_content
                            else:
                                gen_section, gen_usage = llm_gen.generate_section(
                                    suggestion=suggestion,
                                    style=plan.style_profile,
                                    context=context,
                                )
                                if isinstance(gen_section, str) and gen_section:
                                    # Only add header if LLM output doesn't already have one
                                    if not gen_section.strip().startswith('#'):
                                        section_name = e.anchor.get('value', 'section') if e.anchor else 'section'
                                        gen_section = f"## {section_name}\n\n{gen_section}"
                                    e.content_template = gen_section
                    
                    content, stats = renderer.apply_edit(content, e)
                    try:
                        if fpath.endswith((".md", ".rst", ".Rmd", ".Rd")) and content:
                            cleaned, _usage = llm_cleaner.clean_readme(content)
                            if isinstance(cleaned, str) and cleaned.strip():
                                content = cleaned
                    except Exception:
                        pass
                    total_stats["added_lines"] = total_stats.get("added_lines", 0) + stats.get("added_lines", 0)
                
                revised[fpath] = content
                diff_stats[fpath] = total_stats
            
            # Write outputs
            out_repo_key = "satijalab_seurat_fast_test"
            out_dir = output_manager.prepare_output_dir(out_repo_key)
            
            all_files_to_write = dict(files)
            all_files_to_write.update(revised)
            for orig_path, orig_content in files.items():
                all_files_to_write[original_copy_name(orig_path)] = orig_content
            
            artifacts = output_manager.write_files(out_dir, all_files_to_write, diff_stats_by_file=diff_stats)
            
            # Write manifest
            from bioguider.generation.models import GenerationManifest
            manifest = GenerationManifest(
                repo_url=report.repo_url or str(repo_path),
                suggestions=limited_suggestions,
                planned_edits=plan.planned_edits
            )
            output_manager.write_manifest(out_dir, manifest)
            
            # Write generation report using the manager's method
            from bioguider.managers.generation_manager import DocumentationGenerationManager
            import time
            manager = DocumentationGenerationManager(self.llm, None)
            manager.start_time = time.time() - 20.0  # Simulate 20 seconds processing time
            gen_report_path = manager._write_generation_report(
                out_dir,
                report.repo_url or str(repo_path),
                plan,
                diff_stats,
                limited_suggestions,
                artifacts,
                missing,
            )
            
            return out_dir

    try:
        mgr = MockGenerationManager(llm, step_callback)
        mgr.prepare_repo(repo_path)

        out_dir = mgr.run(report_path=report_path, repo_path=repo_path)
        assert os.path.isdir(out_dir)

        manifest_path = os.path.join(out_dir, "manifest.json")
        assert os.path.isfile(manifest_path)

        # Check that generation report exists and shows specific suggestions
        generation_report_path = os.path.join(out_dir, "GENERATION_REPORT.md")
        assert os.path.isfile(generation_report_path)
        

        # Check README if it exists
        readme_path = os.path.join(out_dir, "README.md")
        if os.path.isfile(readme_path):
            with open(readme_path, "r", encoding="utf-8") as fobj:
                content = fobj.read()
                assert len(content) > 0
                assert "Clear 2–3 sentence summary" not in content  # Should not have placeholder content
            readme_original = os.path.join(out_dir, "README.original.md")
            assert os.path.isfile(readme_original)
            
    except Exception as e:
        print(f"Test failed with error: {e}")
        raise

def test_DocumentationGenerationManager_tutorial_section_top10(llm, step_callback):
    """Test that processes only the top 10 RMarkdown (.Rmd) tutorial-related suggestions for quick validation"""
    report_path = "logs/evaluation_report_github_satijalab_seurat.json"
    repo_path = "data/.adalflow/repos/satijalab_seurat"

    # Import the suggestion extractor to limit suggestions
    from bioguider.generation.suggestion_extractor import SuggestionExtractor
    from bioguider.generation.report_loader import EvaluationReportLoader
    from bioguider.generation.change_planner import ChangePlanner
    from bioguider.generation.repo_reader import RepoReader
    from bioguider.generation.style_analyzer import StyleAnalyzer
    import tempfile
    import json
    
    # Load report and extract suggestions
    loader = EvaluationReportLoader()
    report, _ = loader.load(report_path)
    
    extractor = SuggestionExtractor()
    all_suggestions = extractor.extract(report)
    
    # Filter for RMarkdown (.Rmd) tutorial-related suggestions and group by file
    tutorial_keywords = ['tutorial', 'vignette', 'example', 'workflow', 'step-by-step', 'demonstration', 'guide']
    tutorial_suggestions_by_file = {}
    
    for suggestion in all_suggestions:
        # Check if suggestion targets .Rmd files specifically
        target_files = suggestion.target_files or []
        rmd_files = [f for f in target_files if f.lower().endswith('.rmd')]
        
        if not rmd_files:
            continue
            
        # Check if suggestion is tutorial-related
        suggestion_text = str(suggestion.content_guidance or "").lower()
        suggestion_category = str(suggestion.category or "").lower()
        
        # Check if it's tutorial-related by content or category
        is_tutorial = (
            any(keyword in suggestion_text for keyword in tutorial_keywords) or
            any(keyword in suggestion_category for keyword in tutorial_keywords) or
            'tutorial' in suggestion_category
        )
        
        if is_tutorial:
            # Group suggestions by file
            for rmd_file in rmd_files:
                if rmd_file not in tutorial_suggestions_by_file:
                    tutorial_suggestions_by_file[rmd_file] = []
                tutorial_suggestions_by_file[rmd_file].append(suggestion)
                print(f"Found RMarkdown tutorial suggestion: {suggestion.category} -> {rmd_file}")
    
    # Limit to top 3 files with most suggestions (to test multiple suggestions per file)
    sorted_files = sorted(tutorial_suggestions_by_file.items(), key=lambda x: len(x[1]), reverse=True)
    limited_files = sorted_files[:3]  # Top 3 files
    
    # Flatten suggestions from limited files
    limited_suggestions = []
    for file_path, file_suggestions in limited_files:
        limited_suggestions.extend(file_suggestions)
        print(f"File {file_path}: {len(file_suggestions)} suggestions")
    
    print(f"Testing with {len(limited_suggestions)} tutorial suggestions across {len(limited_files)} files (out of {len(tutorial_suggestions_by_file)} files, {len(all_suggestions)} total suggestions)")
    
    if not limited_suggestions:
        print("No tutorial suggestions found, skipping test")
        return
    
    # Read repo files
    reader = RepoReader(repo_path)
    target_files = set()
    for suggestion in limited_suggestions:
        target_files.update(suggestion.target_files)
    
    files, missing = reader.read_files(list(target_files)) if target_files else reader.read_default_targets()
    
    # Analyze style
    style_analyzer = StyleAnalyzer()
    style = style_analyzer.analyze(files)
    
    # Plan changes with limited suggestions
    planner = ChangePlanner()
    plan = planner.build_plan(repo_path=repo_path, style=style, suggestions=limited_suggestions, available_files=files)
    
    print(f"Planned {len(plan.planned_edits)} edits from {len(limited_suggestions)} tutorial suggestions")
    
    # Create a mock generation manager for testing
    class MockTutorialGenerationManager:
        def __init__(self, llm, step_callback):
            self.llm = llm
            self.step_callback = step_callback
            self.loader = loader
            self.extractor = extractor
            self.planner = planner
            self.style_analyzer = style_analyzer
            
        def prepare_repo(self, repo_path):
            pass
            
        def run(self, report_path, repo_path):
            # Use our limited tutorial suggestions instead of extracting from report
            from bioguider.generation.document_renderer import DocumentRenderer
            from bioguider.generation.llm_content_generator import LLMContentGenerator
            from bioguider.generation.llm_cleaner import LLMCleaner
            from bioguider.generation.output_manager import OutputManager
            import os
            
            # Define original_copy_name function locally
            def original_copy_name(path: str) -> str:
                # Handle all file extensions properly
                if "." in path:
                    base, ext = path.rsplit(".", 1)
                    return f"{base}.original.{ext}"
                return f"{path}.original"
            
            # Initialize components
            renderer = DocumentRenderer()
            llm_gen = LLMContentGenerator(self.llm)
            llm_cleaner = LLMCleaner(self.llm)
            output_manager = OutputManager()
            
            # Use our pre-planned edits
            revised = {}
            diff_stats = {}
            edits_by_file = {}
            for e in plan.planned_edits:
                edits_by_file.setdefault(e.file_path, []).append(e)
            
            # Process each file
            for fpath, edits in edits_by_file.items():
                original_content = files.get(fpath, "")
                content = original_content
                total_stats = {"added_lines": 0}
                
                # Group suggestions by file to avoid duplicate generation
                file_suggestions = []
                full_replace_edits = []
                section_edits = []
                
                for e in edits:
                    suggestion = next((s for s in limited_suggestions if s.id == e.suggestion_id), None) if e.suggestion_id else None
                    if suggestion:
                        file_suggestions.append(suggestion)
                        if e.edit_type == "full_replace":
                            full_replace_edits.append(e)
                        else:
                            section_edits.append(e)
                
                # Generate content once per file if there are full_replace edits
                if full_replace_edits:
                    print(f"Generating full document for {fpath} with {len(file_suggestions)} suggestions using LLM...")
                    
                    # Merge all suggestions for this file into a single evaluation report
                    merged_evaluation_report = {
                        "suggestions": [{"content_guidance": s.content_guidance, "evidence": s.source.get("evidence", "") if s.source else ""} for s in file_suggestions],
                        "total_suggestions": len(file_suggestions)
                    }
                    
                    gen_content, gen_usage = llm_gen.generate_full_document(
                        target_file=fpath,
                        evaluation_report=merged_evaluation_report,
                        context=original_content,
                    )
                    
                    if isinstance(gen_content, str) and gen_content:
                        print(f"✓ Generated full document for {fpath} ({gen_usage.get('total_tokens', 0)} tokens)")
                        # Apply the generated content to all full_replace edits
                        for e in full_replace_edits:
                            e.content_template = gen_content
                        content = gen_content
                    else:
                        # Fallback: try individual generation but only for the first edit to avoid duplicates
                        if full_replace_edits:
                            e = full_replace_edits[0]  # Only process the first edit
                            suggestion = next((s for s in limited_suggestions if s.id == e.suggestion_id), None) if e.suggestion_id else None
                            if suggestion and (not e.content_template or e.content_template.strip() == ""):
                                print(f"Fallback: Generating full document for {e.suggestion_id} using LLM...")
                                gen_content, gen_usage = llm_gen.generate_full_document(
                                    target_file=e.file_path,
                                    evaluation_report={"suggestion": suggestion.content_guidance, "evidence": suggestion.source.get("evidence", "") if suggestion.source else ""},
                                    context=original_content,
                                )
                                if isinstance(gen_content, str) and gen_content:
                                    print(f"✓ Generated full document for {e.suggestion_id} ({gen_usage.get('total_tokens', 0)} tokens)")
                                    # Apply the same content to all full_replace edits
                                    for edit in full_replace_edits:
                                        edit.content_template = gen_content
                                    content = gen_content
                else:
                    # Handle section edits individually
                    for e in section_edits:
                        suggestion = next((s for s in limited_suggestions if s.id == e.suggestion_id), None) if e.suggestion_id else None
                        if suggestion and (not e.content_template or e.content_template.strip() == ""):
                            print(f"Generating section for {e.suggestion_id} using LLM...")
                            gen_section, gen_usage = llm_gen.generate_section(
                                suggestion=suggestion,
                                style=plan.style_profile,
                                context=original_content,
                            )
                            if isinstance(gen_section, str) and gen_section:
                                print(f"✓ Generated section for {e.suggestion_id} ({gen_usage.get('total_tokens', 0)} tokens)")
                                # Ensure header present
                                if gen_section.lstrip().startswith("#"):
                                    e.content_template = gen_section
                                else:
                                    title = e.anchor.get('value', '').strip() or ''
                                    e.content_template = f"## {title}\n\n{gen_section}" if title else gen_section
                        
                        content, stats = renderer.apply_edit(content, e)
                        total_stats["added_lines"] = total_stats.get("added_lines", 0) + stats.get("added_lines", 0)
                
                # Apply remaining edits that weren't full_replace
                for e in edits:
                    if e.edit_type != "full_replace":
                        content, stats = renderer.apply_edit(content, e)
                        total_stats["added_lines"] = total_stats.get("added_lines", 0) + stats.get("added_lines", 0)
                
                # Clean content
                try:
                    if fpath.endswith((".md", ".rst", ".Rmd", ".Rd")) and content:
                        cleaned, _usage = llm_cleaner.clean_readme(content)
                        if isinstance(cleaned, str) and cleaned.strip():
                            content = cleaned
                except Exception:
                    pass
                
                revised[fpath] = content
                diff_stats[fpath] = total_stats
                print(f"✓ Completed {fpath} - added {total_stats['added_lines']} lines")
            
            # Write outputs
            out_repo_key = "satijalab_seurat_tutorial_test"
            out_dir = output_manager.prepare_output_dir(out_repo_key)
            
            all_files_to_write = dict(files)
            all_files_to_write.update(revised)
            for orig_path, orig_content in files.items():
                all_files_to_write[original_copy_name(orig_path)] = orig_content
            
            artifacts = output_manager.write_files(out_dir, all_files_to_write, diff_stats_by_file=diff_stats)
            
            # Write manifest
            from bioguider.generation.models import GenerationManifest
            manifest = GenerationManifest(
                repo_url=report.repo_url or str(repo_path),
                suggestions=limited_suggestions,
                planned_edits=plan.planned_edits
            )
            output_manager.write_manifest(out_dir, manifest)
            
            # Write generation report using the manager's method
            from bioguider.managers.generation_manager import DocumentationGenerationManager
            import time
            manager = DocumentationGenerationManager(self.llm, None)
            manager.start_time = time.time() - 20.0  # Simulate 20 seconds processing time
            gen_report_path = manager._write_generation_report(
                out_dir,
                report.repo_url or str(repo_path),
                plan,
                diff_stats,
                limited_suggestions,
                artifacts,
                missing,
            )
            
            return out_dir

    try:
        mgr = MockTutorialGenerationManager(llm, step_callback)
        mgr.prepare_repo(repo_path)

        out_dir = mgr.run(report_path=report_path, repo_path=repo_path)
        assert os.path.isdir(out_dir)

        manifest_path = os.path.join(out_dir, "manifest.json")
        assert os.path.isfile(manifest_path)

        # Check that generation report exists and shows tutorial suggestions
        generation_report_path = os.path.join(out_dir, "GENERATION_REPORT.md")
        assert os.path.isfile(generation_report_path)
        
        # Check for tutorial files specifically
        tutorial_files = [f for f in os.listdir(out_dir) if f.endswith('.Rmd') or 'tutorial' in f.lower() or 'vignette' in f.lower()]
        if tutorial_files:
            print(f"Found tutorial files: {tutorial_files}")
            for tutorial_file in tutorial_files:
                tutorial_path = os.path.join(out_dir, tutorial_file)
                if os.path.isfile(tutorial_path):
                    with open(tutorial_path, "r", encoding="utf-8") as fobj:
                        content = fobj.read()
                        assert len(content) > 0
                        print(f"✓ Tutorial file {tutorial_file}: {len(content)} characters")

        # Check README if it exists
        readme_path = os.path.join(out_dir, "README.md")
        if os.path.isfile(readme_path):
            with open(readme_path, "r", encoding="utf-8") as fobj:
                content = fobj.read()
                assert len(content) > 0
                assert "Clear 2–3 sentence summary" not in content  # Should not have placeholder content
            readme_original = os.path.join(out_dir, "README.original.md")
            assert os.path.isfile(readme_original)
            
    except Exception as e:
        print(f"Tutorial test failed with error: {e}")
        raise

@pytest.mark.skip(reason="deprecated")
def test_DocumentationGenerationManager_on_biogsp_newformat(llm, step_callback):
    report_path = "logs/evaluation_report_github_fengsh27_BioGSP.json"
    repo_path = "data/.adalflow/repos/BioGSP"

    mgr = DocumentationGenerationManager(llm, step_callback)
    mgr.prepare_repo(repo_path)

    out_dir = mgr.run(report_path=report_path, repo_path=repo_path)
    assert os.path.isdir(out_dir)

    manifest_path = os.path.join(out_dir, "manifest.json")
    assert os.path.isfile(manifest_path)

    readme_path = os.path.join(out_dir, "README.md")
    if os.path.isfile(readme_path):
        with open(readme_path, "r", encoding="utf-8") as fobj:
            content = fobj.read()
            assert len(content) > 0
        readme_original = os.path.join(out_dir, "README.original.md")
        assert os.path.isfile(readme_original)
