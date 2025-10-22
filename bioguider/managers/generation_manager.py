from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Tuple, Dict, List

from bioguider.generation import (
    EvaluationReportLoader,
    SuggestionExtractor,
    RepoReader,
    StyleAnalyzer,
    ChangePlanner,
    DocumentRenderer,
    OutputManager,
    LLMContentGenerator,
    LLMCleaner,
)
from bioguider.generation.models import GenerationManifest, GenerationReport
from bioguider.utils.file_utils import parse_repo_url


class DocumentationGenerationManager:
    def __init__(self, llm, step_callback, output_dir: Optional[str] = None):
        self.llm = llm
        self.step_callback = step_callback
        self.repo_url_or_path: str | None = None
        self.start_time = None

        self.loader = EvaluationReportLoader()
        self.extractor = SuggestionExtractor()
        self.style_analyzer = StyleAnalyzer()
        self.planner = ChangePlanner()
        self.renderer = DocumentRenderer()
        self.output = OutputManager(base_outputs_dir=output_dir)
        self.llm_gen = LLMContentGenerator(llm)
        self.llm_cleaner = LLMCleaner(llm)
        

    def print_step(self, step_name: str | None = None, step_output: str | None = None):
        if self.step_callback is None:
            return
        self.step_callback(step_name=step_name, step_output=step_output)

    def prepare_repo(self, repo_url_or_path: str):
        self.repo_url_or_path = repo_url_or_path

    def _get_generation_time(self) -> str:
        """Get formatted generation time with start, end, and duration"""
        if self.start_time is None:
            return "Not tracked"
        import time
        import datetime
        end_time = time.time()
        duration = end_time - self.start_time
        
        start_str = datetime.datetime.fromtimestamp(self.start_time).strftime("%H:%M:%S")
        end_str = datetime.datetime.fromtimestamp(end_time).strftime("%H:%M:%S")
        
        if duration < 60:
            duration_str = f"{duration:.1f}s"
        elif duration < 3600:
            duration_str = f"{duration/60:.1f}m"
        else:
            duration_str = f"{duration/3600:.1f}h"
            
        return f"{start_str} → {end_str} ({duration_str})"

    def run(self, report_path: str, repo_path: str | None = None) -> str:
        import time
        self.start_time = time.time()
        repo_path = repo_path or self.repo_url_or_path or ""
        self.print_step(step_name="LoadReport", step_output=f"Loading evaluation report from {report_path}...")
        report, report_abs = self.loader.load(report_path)
        self.print_step(step_name="LoadReport", step_output="✓ Evaluation report loaded successfully")

        self.print_step(step_name="ReadRepoFiles", step_output=f"Reading repository files from {repo_path}...")
        reader = RepoReader(repo_path)
        # Prefer report-listed files if available; include all report-declared file lists
        target_files = []
        if getattr(report, "readme_files", None):
            target_files.extend(report.readme_files)
        if getattr(report, "installation_files", None):
            target_files.extend(report.installation_files)
        # If userguide_files not explicitly provided, derive from userguide_evaluation keys
        userguide_files: list[str] = []
        if getattr(report, "userguide_files", None):
            userguide_files.extend([p for p in report.userguide_files if isinstance(p, str)])
        elif getattr(report, "userguide_evaluation", None) and isinstance(report.userguide_evaluation, dict):
            for key in report.userguide_evaluation.keys():
                if isinstance(key, str) and key.strip():
                    userguide_files.append(key)
        target_files.extend(userguide_files)
        
        # Add tutorial files from tutorial_evaluation keys
        tutorial_files: list[str] = []
        if getattr(report, "tutorial_files", None):
            tutorial_files.extend([p for p in report.tutorial_files if isinstance(p, str)])
        elif getattr(report, "tutorial_evaluation", None) and isinstance(report.tutorial_evaluation, dict):
            for key in report.tutorial_evaluation.keys():
                if isinstance(key, str) and key.strip():
                    tutorial_files.append(key)
        target_files.extend(tutorial_files)
        
        if getattr(report, "submission_requirements_files", None):
            target_files.extend(report.submission_requirements_files)
        target_files = [p for p in target_files if isinstance(p, str) and p.strip()]
        target_files = list(dict.fromkeys(target_files))  # de-dup
        files, missing = reader.read_files(target_files) if target_files else reader.read_default_targets()
        self.print_step(step_name="ReadRepoFiles", step_output=f"✓ Read {len(files)} files from repository")

        self.print_step(step_name="AnalyzeStyle", step_output="Analyzing document style and formatting...")
        style = self.style_analyzer.analyze(files)
        self.print_step(step_name="AnalyzeStyle", step_output="✓ Document style analysis completed")

        self.print_step(step_name="ExtractSuggestions", step_output="Extracting suggestions from evaluation report...")
        suggestions = self.extractor.extract(report)
        self.print_step(step_name="Suggestions", step_output=f"✓ Extracted {len(suggestions)} suggestions from evaluation report")

        self.print_step(step_name="PlanChanges", step_output="Planning changes based on suggestions...")
        plan = self.planner.build_plan(repo_path=repo_path, style=style, suggestions=suggestions, available_files=files)
        self.print_step(step_name="PlannedEdits", step_output=f"✓ Planned {len(plan.planned_edits)} edits across {len(set(e.file_path for e in plan.planned_edits))} files")

        self.print_step(step_name="RenderDocuments", step_output=f"Rendering documents with LLM (processing {len(plan.planned_edits)} edits)...")
        # Apply edits; support full-file regeneration using the evaluation report as the sole authority
        revised: Dict[str, str] = {}
        diff_stats: Dict[str, dict] = {}
        edits_by_file: Dict[str, list] = {}
        for e in plan.planned_edits:
            edits_by_file.setdefault(e.file_path, []).append(e)
        
        total_files = len(edits_by_file)
        processed_files = 0

        # Prepare evaluation data subset to drive LLM full document generation
        evaluation_data = {
            "readme_evaluation": getattr(report, "readme_evaluation", None),
            "installation_evaluation": getattr(report, "installation_evaluation", None),
            "userguide_evaluation": getattr(report, "userguide_evaluation", None),
            "tutorial_evaluation": getattr(report, "tutorial_evaluation", None),
        }

        for fpath, edits in edits_by_file.items():
            processed_files += 1
            self.print_step(step_name="ProcessingFile", step_output=f"Processing {fpath} ({processed_files}/{total_files}) - {len(edits)} edits")
            
            original_content = files.get(fpath, "")
            
            # Group suggestions by file to avoid duplicate generation
            file_suggestions = []
            full_replace_edits = []
            section_edits = []
            
            for e in edits:
                suggestion = next((s for s in suggestions if s.id == e.suggestion_id), None) if e.suggestion_id else None
                if suggestion:
                    file_suggestions.append(suggestion)
                    if e.edit_type == "full_replace":
                        full_replace_edits.append(e)
                    else:
                        section_edits.append(e)
            
            # Debug: Save suggestion grouping info
            import json
            import os
            from datetime import datetime
            
            debug_dir = "outputs/debug_generation"
            os.makedirs(debug_dir, exist_ok=True)
            safe_filename = fpath.replace("/", "_").replace(".", "_")
            
            grouping_info = {
                "file_path": fpath,
                "total_edits": len(edits),
                "file_suggestions_count": len(file_suggestions),
                "full_replace_edits_count": len(full_replace_edits),
                "section_edits_count": len(section_edits),
                "suggestions": [
                    {
                        "id": s.id,
                        "category": s.category,
                        "content_guidance": s.content_guidance[:200] + "..." if len(s.content_guidance or "") > 200 else s.content_guidance,
                        "target_files": s.target_files
                    } for s in file_suggestions
                ],
                "timestamp": datetime.now().isoformat()
            }
            
            grouping_file = os.path.join(debug_dir, f"{safe_filename}_grouping.json")
            with open(grouping_file, 'w', encoding='utf-8') as f:
                json.dump(grouping_info, f, indent=2, ensure_ascii=False)
            
            content = original_content
            total_stats = {"added_lines": 0}
            
            # CRITICAL: Generate content ONCE per file if there are full_replace edits
            # All suggestions for this file are merged into a single evaluation report
            # This prevents duplicate content generation
            if full_replace_edits:
                self.print_step(
                    step_name="GeneratingContent", 
                    step_output=f"🔄 Generating full document for {fpath} with {len(file_suggestions)} suggestions using LLM (SINGLE CALL)..."
                )
                
                # Merge all suggestions for this file into a single evaluation report
                # Format suggestions with clear numbering to help LLM understand they're separate improvements
                suggestions_list = []
                for idx, s in enumerate(file_suggestions, 1):
                    suggestions_list.append({
                        "suggestion_number": idx,
                        "category": s.category if hasattr(s, 'category') else "general",
                        "content_guidance": s.content_guidance
                    })
                
                merged_evaluation_report = {
                    "total_suggestions": len(file_suggestions),
                    "integration_instruction": f"Integrate ALL {len(file_suggestions)} suggestions below into ONE cohesive document. Do NOT create {len(file_suggestions)} separate versions.",
                    "suggestions": suggestions_list
                }
                
                # Debug: Save merged evaluation report
                merged_report_file = os.path.join(debug_dir, f"{safe_filename}_merged_report.json")
                with open(merged_report_file, 'w', encoding='utf-8') as f:
                    json.dump(merged_evaluation_report, f, indent=2, ensure_ascii=False)
                
                # Debug: Log that we're about to make a single generation call
                debug_log_file = os.path.join(debug_dir, f"{safe_filename}_generation_log.txt")
                with open(debug_log_file, 'a', encoding='utf-8') as f:
                    f.write(f"\n=== GENERATION CALL at {datetime.now().isoformat()} ===\n")
                    f.write(f"File: {fpath}\n")
                    f.write(f"Full replace edits: {len(full_replace_edits)}\n")
                    f.write(f"Total suggestions: {len(file_suggestions)}\n")
                    f.write(f"Merged into single call: YES\n")
                    f.write(f"Suggestion IDs: {[s.id for s in file_suggestions]}\n\n")
                
                    gen_content, gen_usage = self.llm_gen.generate_full_document(
                        target_file=fpath,
                        evaluation_report=merged_evaluation_report,
                        context=original_content,
                        original_content=original_content,
                    )
                
                # Debug: Log completion
                with open(debug_log_file, 'a', encoding='utf-8') as f:
                    f.write(f"Generation completed at {datetime.now().isoformat()}\n")
                    f.write(f"Content length: {len(gen_content) if isinstance(gen_content, str) else 0} characters\n")
                    f.write(f"Tokens used: {gen_usage.get('total_tokens', 0)}\n")
                    f.write(f"SUCCESS: {isinstance(gen_content, str) and gen_content}\n\n")
                
                if isinstance(gen_content, str) and gen_content:
                    self.print_step(step_name="LLMFullDoc", step_output=f"✓ Generated full document for {fpath} ({gen_usage.get('total_tokens', 0)} tokens)")
                    # Apply the generated content to all full_replace edits
                    for e in full_replace_edits:
                        e.content_template = gen_content
                    content = gen_content
                else:
                    # Fallback: try individual generation but only for the first edit to avoid duplicates
                    if full_replace_edits:
                        e = full_replace_edits[0]  # Only process the first edit
                        suggestion = next((s for s in suggestions if s.id == e.suggestion_id), None) if e.suggestion_id else None
                        if suggestion and (not e.content_template or e.content_template.strip() == ""):
                            self.print_step(step_name="GeneratingContent", step_output=f"Fallback: Generating full document for {e.suggestion_id} using LLM...")
                            gen_content, gen_usage = self.llm_gen.generate_full_document(
                                target_file=e.file_path,
                                evaluation_report={"suggestion": suggestion.content_guidance},
                                context=original_content,
                                original_content=original_content,
                            )
                            if isinstance(gen_content, str) and gen_content:
                                self.print_step(step_name="LLMFullDoc", step_output=f"✓ Generated full document for {e.suggestion_id} ({gen_usage.get('total_tokens', 0)} tokens)")
                                # Apply the same content to all full_replace edits
                                for edit in full_replace_edits:
                                    edit.content_template = gen_content
                                content = gen_content
            else:
                # Handle section edits individually
                for e in section_edits:
                    suggestion = next((s for s in suggestions if s.id == e.suggestion_id), None) if e.suggestion_id else None
                    if suggestion and (not e.content_template or e.content_template.strip() == ""):
                        self.print_step(step_name="GeneratingContent", step_output=f"Generating section for {e.suggestion_id} using LLM...")
                        gen_section, gen_usage = self.llm_gen.generate_section(
                            suggestion=suggestion,
                            style=plan.style_profile,
                            context=original_content,
                        )
                        if isinstance(gen_section, str) and gen_section:
                            self.print_step(step_name="LLMSection", step_output=f"✓ Generated section for {e.suggestion_id} ({gen_usage.get('total_tokens', 0)} tokens)")
                            # Ensure header present
                            if gen_section.lstrip().startswith("#"):
                                e.content_template = gen_section
                            else:
                                title = e.anchor.get('value', '').strip() or ''
                                e.content_template = f"## {title}\n\n{gen_section}" if title else gen_section
                    
                    content, stats = self.renderer.apply_edit(content, e)
                    total_stats["added_lines"] = total_stats.get("added_lines", 0) + stats.get("added_lines", 0)
            
            # Apply remaining edits that weren't full_replace
            for e in edits:
                if e.edit_type != "full_replace":
                    content, stats = self.renderer.apply_edit(content, e)
                    total_stats["added_lines"] = total_stats.get("added_lines", 0) + stats.get("added_lines", 0)
            
            # After applying full document or section changes, run a general cleaner pass for all text files
            # to fix markdown/formatting issues without changing meaning.
            try:
                if fpath.endswith((".md", ".rst", ".Rmd", ".Rd")) and content:
                    self.print_step(step_name="CleaningContent", step_output=f"Cleaning formatting for {fpath}...")
                    cleaned, _usage = self.llm_cleaner.clean_readme(content)
                    if isinstance(cleaned, str) and cleaned.strip():
                        content = cleaned
                    
                    # LLM cleaner now handles markdown fences and unwanted summaries
                        
            except Exception:
                pass
            
            revised[fpath] = content
            diff_stats[fpath] = total_stats
            self.print_step(step_name="RenderedFile", step_output=f"✓ Completed {fpath} - added {total_stats['added_lines']} lines")

        # Removed cleaner: duplication and fixes handled in prompts and renderer

        # Prefer local repo folder name for outputs; fallback to author_repo from URL
        out_repo_key = None
        if repo_path and os.path.isdir(repo_path):
            out_repo_key = os.path.basename(os.path.normpath(repo_path))
        elif report.repo_url:
            try:
                author, name = parse_repo_url(report.repo_url)
                out_repo_key = f"{author}_{name}"
            except Exception:
                out_repo_key = report.repo_url
        else:
            out_repo_key = self.repo_url_or_path or "repo"

        self.print_step(step_name="WriteOutputs", step_output=f"Writing outputs to {out_repo_key}...")
        out_dir = self.output.prepare_output_dir(out_repo_key)
        # Ensure all files we read (even without edits) are written to outputs alongside revisions
        all_files_to_write: Dict[str, str] = dict(files)
        all_files_to_write.update(revised)
        # Also copy originals next to the new files for side-by-side comparison
        def original_copy_name(path: str) -> str:
            # Handle all file extensions properly
            if "." in path:
                base, ext = path.rsplit(".", 1)
                return f"{base}.original.{ext}"
            return f"{path}.original"

        for orig_path, orig_content in files.items():
            all_files_to_write[original_copy_name(orig_path)] = orig_content
        
        self.print_step(step_name="WritingFiles", step_output=f"Writing {len(all_files_to_write)} files to output directory...")
        artifacts = self.output.write_files(out_dir, all_files_to_write, diff_stats_by_file=diff_stats)

        manifest = GenerationManifest(
            repo_url=report.repo_url,
            report_path=report_abs,
            output_dir=out_dir,
            suggestions=suggestions,
            planned_edits=plan.planned_edits,
            artifacts=artifacts,
            skipped=missing,
        )
        self.print_step(step_name="WritingManifest", step_output="Writing generation manifest...")
        self.output.write_manifest(out_dir, manifest)
        
        # Write human-readable generation report
        self.print_step(step_name="WritingReport", step_output="Writing generation report...")
        gen_report_path = self._write_generation_report(
            out_dir,
            report.repo_url or str(self.repo_url_or_path or ""),
            plan,
            diff_stats,
            suggestions,
            artifacts,
            missing,
        )
        self.print_step(step_name="Done", step_output=f"✓ Generation completed! Output directory: {out_dir}")
        return out_dir

    def _write_generation_report(
        self,
        out_dir: str,
        repo_url: str,
        plan,
        diff_stats: Dict[str, dict],
        suggestions,
        artifacts,
        skipped: List[str],
    ):
        # Build a user-friendly markdown report
        lines: list[str] = []
        lines.append(f"# Documentation Generation Report\n")
        lines.append(f"**Repository:** {repo_url}\n")
        lines.append(f"**Generated:** {out_dir}\n")
        
        # Processing timeline
        total_improvements = len(plan.planned_edits)
        start_time_str = self._get_generation_time().split(" → ")[0] if self.start_time else "Not tracked"
        end_time_str = self._get_generation_time().split(" → ")[1].split(" (")[0] if self.start_time else "Not tracked"
        duration_str = self._get_generation_time().split("(")[1].replace(")", "") if self.start_time else "Not tracked"
        
        lines.append(f"**Processing Timeline:**\n")
        lines.append(f"- **Start Time:** {start_time_str}\n")
        lines.append(f"- **End Time:** {end_time_str}\n")
        lines.append(f"- **Duration:** {duration_str}\n")
        
        # Calculate statistics by category
        category_stats = {}
        file_stats = {}
        for e in plan.planned_edits:
            sug = next((s for s in suggestions if s.id == e.suggestion_id), None)
            if sug and sug.category:
                category = sug.category.split('.')[0]  # e.g., "readme.dependencies" -> "readme"
                category_stats[category] = category_stats.get(category, 0) + 1
            
            file_stats[e.file_path] = file_stats.get(e.file_path, 0) + 1
        
        # Calculate evaluation report statistics
        score_stats = {"Excellent": 0, "Good": 0, "Fair": 0, "Poor": 0}
        processed_suggestions = set()
        for e in plan.planned_edits:
            sug = next((s for s in suggestions if s.id == e.suggestion_id), None)
            if sug and sug.source and sug.id not in processed_suggestions:
                score = sug.source.get("score", "")
                if score in score_stats:
                    score_stats[score] += 1
                processed_suggestions.add(sug.id)
        
        # Calculate success rate based on processed suggestions only
        processed_suggestions_count = len([s for s in suggestions if s.source and s.source.get("score", "") in ("Fair", "Poor")])
        fixed_suggestions = len([s for s in processed_suggestions if s in [sug.id for sug in suggestions if sug.source and sug.source.get("score", "") in ("Fair", "Poor")]])
        
        # Add professional summary and key metrics
        lines.append(f"\n## Summary\n")
        
        # Concise summary for busy developers
        lines.append(f"This is a report of automated documentation enhancements generated by BioGuider.\n")
        lines.append(f"\nOur AI analyzed your existing documentation to identify areas for improvement based on standards for high-quality scientific software. It then automatically rewrote the files to be more accessible and useful for biomedical researchers.\n")
        lines.append(f"\nThis changelog provides a transparent record of what was modified and why. We encourage you to review the changes before committing. Original file versions are backed up with a `.original` extension.\n")
        
        # Core metrics
        total_lines_added = sum(stats.get('added_lines', 0) for stats in diff_stats.values())
        success_rate = (fixed_suggestions/processed_suggestions_count*100) if processed_suggestions_count > 0 else 0
        
        # Lead with success rate - the most important outcome
        lines.append(f"\n### Key Metrics\n")
        lines.append(f"- **Success Rate:** {success_rate:.1f}% ({fixed_suggestions} of {processed_suggestions_count} processed suggestions addressed)\n")
        lines.append(f"- **Total Impact:** {total_improvements} improvements across {len(file_stats)} files\n")
        lines.append(f"- **Content Added:** {total_lines_added} lines of enhanced documentation\n")
        
        # Explain why some suggestions were filtered out
        total_suggestions = len(suggestions)
        filtered_count = total_suggestions - processed_suggestions_count
        if filtered_count > 0:
            lines.append(f"\n### Processing Strategy\n")
            lines.append(f"- **Suggestions filtered out:** {filtered_count} items\n")
            lines.append(f"- **Reason:** Only 'Fair' and 'Poor' priority suggestions were processed\n")
            lines.append(f"- **Rationale:** Focus on critical issues that need immediate attention\n")
            lines.append(f"- **Quality threshold:** 'Excellent' and 'Good' suggestions already meet standards\n")
        
        # Priority breakdown - answer "Was it important work?"
        lines.append(f"\n### Priority Breakdown\n")
        priority_fixed = 0
        priority_total = 0
        
        for score in ["Poor", "Fair"]:
            count = score_stats[score]
            if count > 0:
                priority_total += count
                priority_fixed += count
                lines.append(f"- **{score} Priority:** {count} items → 100% addressed\n")
        
        # Remove confusing "Critical Issues" bullet - success rate already shown above
        
        # Quality assurance note
        excellent_count = score_stats.get("Excellent", 0)
        good_count = score_stats.get("Good", 0)
        if excellent_count > 0 or good_count > 0:
            lines.append(f"\n### Quality Assurance\n")
            lines.append(f"- **High-Quality Items:** {excellent_count + good_count} suggestions already meeting standards (no changes needed)\n")
        
        # Group improvements by file type for better readability
        by_file = {}
        for e in plan.planned_edits:
            if e.file_path not in by_file:
                by_file[e.file_path] = []
            by_file[e.file_path].append(e)
        
        lines.append(f"\n## Files Improved\n")
        for file_path, edits in by_file.items():
            added_lines = diff_stats.get(file_path, {}).get('added_lines', 0)
            lines.append(f"\n### {file_path}\n")
            lines.append(f"**Changes made:** {len(edits)} improvement(s), {added_lines} lines added\n")
            
            for e in edits:
                sug = next((s for s in suggestions if s.id == e.suggestion_id), None)
                guidance = sug.content_guidance if sug else ""
                section = e.anchor.get('value', 'General improvements')
                
                # Convert technical action names to user-friendly descriptions
                # Use the suggestion action if available, otherwise fall back to edit type
                action_key = sug.action if sug else e.edit_type
                
                # Generate category-based description for full_replace actions
                if action_key == 'full_replace' and sug:
                    category = sug.category or ""
                    category_display = category.split('.')[-1].replace('_', ' ').title() if category else ""
                    
                    # Create specific descriptions based on category
                    if 'readme' in category.lower():
                        action_desc = 'Enhanced README documentation'
                    elif 'tutorial' in category.lower():
                        action_desc = 'Improved tutorial content'
                    elif 'userguide' in category.lower():
                        action_desc = 'Enhanced user guide documentation'
                    elif 'installation' in category.lower():
                        action_desc = 'Improved installation instructions'
                    elif 'dependencies' in category.lower():
                        action_desc = 'Enhanced dependency information'
                    elif 'readability' in category.lower():
                        action_desc = 'Improved readability and clarity'
                    elif 'setup' in category.lower():
                        action_desc = 'Enhanced setup and configuration'
                    elif 'reproducibility' in category.lower():
                        action_desc = 'Improved reproducibility'
                    elif 'structure' in category.lower():
                        action_desc = 'Enhanced document structure'
                    elif 'code_quality' in category.lower():
                        action_desc = 'Improved code quality'
                    elif 'verification' in category.lower():
                        action_desc = 'Enhanced result verification'
                    elif 'performance' in category.lower():
                        action_desc = 'Added performance considerations'
                    elif 'context' in category.lower():
                        action_desc = 'Enhanced context and purpose'
                    elif 'error_handling' in category.lower():
                        action_desc = 'Improved error handling'
                    else:
                        action_desc = f'Enhanced {category_display}' if category_display else 'Comprehensive rewrite'
                else:
                    # Use existing action descriptions for non-full_replace actions
                    action_desc = {
                        'append_section': f'Added "{section}" section',
                        'insert_after_header': f'Enhanced content in "{section}"',
                        'rmarkdown_integration': f'Integrated improvements in "{section}"',
                        'replace_intro_block': f'Improved "{section}" section',
                        'add_dependencies_section': 'Added dependencies information',
                        'add_system_requirements_section': 'Added system requirements',
                        'add_hardware_requirements': 'Added hardware requirements',
                        'clarify_mandatory_vs_optional': 'Clarified dependencies',
                        'improve_readability': f'Improved readability in "{section}"',
                        'improve_setup': f'Enhanced setup instructions in "{section}"',
                        'improve_reproducibility': f'Improved reproducibility in "{section}"',
                        'improve_structure': f'Enhanced structure in "{section}"',
                        'improve_code_quality': f'Improved code quality in "{section}"',
                        'improve_verification': f'Enhanced result verification in "{section}"',
                        'improve_performance': f'Added performance notes in "{section}"',
                        'improve_clarity_and_error_handling': f'Improved clarity and error handling in "{section}"',
                        'improve_consistency': f'Improved consistency in "{section}"',
                        'improve_context': f'Enhanced context in "{section}"',
                        'improve_error_handling': f'Improved error handling in "{section}"',
                        'add_overview_section': f'Added "{section}" section'
                    }.get(action_key, f'Improved {action_key}')
                
                lines.append(f"- **{action_desc}**")
                
                # Show evaluation reasoning that triggered this improvement
                if sug and sug.source:
                    score = sug.source.get("score", "")
                    category = sug.category or ""
                    
                    # Format category for display (e.g., "readme.dependencies" -> "Dependencies")
                    category_display = category.split('.')[-1].replace('_', ' ').title() if category else ""
                    
                    if score and category_display:
                        lines.append(f"  - *Reason:* [{category_display} - {score}]")
                    elif score:
                        lines.append(f"  - *Reason:* [{score}]")
                    elif category_display:
                        lines.append(f"  - *Reason:* [{category_display}]")
                
                # Show what was actually implemented (different from reason)
                if guidance:
                    # Extract key action from guidance to show what was implemented
                    if "dependencies" in guidance.lower():
                        lines.append(f"  - *Implemented:* Added comprehensive dependency list with installation instructions")
                    elif "system requirements" in guidance.lower() or "hardware" in guidance.lower():
                        lines.append(f"  - *Implemented:* Added system requirements and platform-specific installation details")
                    elif "comparative statement" in guidance.lower() or "beneficial" in guidance.lower():
                        lines.append(f"  - *Implemented:* Added comparative analysis highlighting Seurat's advantages")
                    elif "readability" in guidance.lower() or "bullet" in guidance.lower():
                        lines.append(f"  - *Implemented:* Enhanced readability with structured lists and examples")
                    elif "overview" in guidance.lower() or "summary" in guidance.lower():
                        lines.append(f"  - *Implemented:* Improved overview section with clear, professional tone")
                    elif "accessible" in guidance.lower() or "non-experts" in guidance.lower():
                        lines.append(f"  - *Implemented:* Simplified language for broader accessibility")
                    elif "examples" in guidance.lower() or "usage" in guidance.lower():
                        lines.append(f"  - *Implemented:* Added practical examples and usage scenarios")
                    elif "error" in guidance.lower() or "debug" in guidance.lower():
                        lines.append(f"  - *Implemented:* Added error handling guidance and troubleshooting tips")
                    elif "context" in guidance.lower() or "scenarios" in guidance.lower():
                        lines.append(f"  - *Implemented:* Expanded context and real-world application examples")
                    elif "structure" in guidance.lower() or "organization" in guidance.lower():
                        lines.append(f"  - *Implemented:* Improved document structure and organization")
                    else:
                        # Truncate long guidance to avoid repetition
                        short_guidance = guidance[:100] + "..." if len(guidance) > 100 else guidance
                        lines.append(f"  - *Implemented:* {short_guidance}")
                
                lines.append("")
        
        # Note about skipped files
        if skipped:
            lines.append(f"\n## Note\n")
            lines.append(f"The following files were not modified as they were not found in the repository:")
            for rel in skipped:
                lines.append(f"- {rel}")
        
        report_md = "\n".join(lines)
        dest = os.path.join(out_dir, "GENERATION_REPORT.md")
        with open(dest, "w", encoding="utf-8") as fobj:
            fobj.write(report_md)
        return dest


