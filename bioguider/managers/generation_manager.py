"""
Documentation Generation Manager - orchestrates the document improvement pipeline.

Refactored from the original monolithic implementation to use smaller, focused methods.
"""

from __future__ import annotations

import os
import time
import datetime
import json
import logging
from pathlib import Path
from typing import Optional, Tuple, Dict, List, Any

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
from bioguider.generation.models import (
    GenerationManifest,
    GenerationReport,
    EvaluationReport,
    SuggestionItem,
    PlannedEdit,
    DocumentPlan,
    StyleProfile,
)
from bioguider.utils.file_utils import parse_repo_url
from bioguider.managers.config import GenerationConfig


logger = logging.getLogger(__name__)


class DocumentationGenerationManager:
    """
    Orchestrates the documentation generation pipeline.

    Pipeline steps:
    1. Load evaluation report
    2. Read repository files
    3. Analyze document style
    4. Extract improvement suggestions
    5. Plan changes
    6. Render/generate content with LLM
    7. Write outputs
    """

    def __init__(
        self,
        llm,
        step_callback,
        output_dir: Optional[str] = None,
        config: Optional[GenerationConfig] = None,
    ):
        """
        Initialize the generation manager.

        Args:
            llm: Language model for content generation
            step_callback: Callback for progress reporting
            output_dir: Optional base directory for outputs
            config: Optional configuration object
        """
        self.llm = llm
        self.step_callback = step_callback
        self.repo_url_or_path: str | None = None
        self.start_time: float | None = None
        self.config = config or GenerationConfig(output_dir=output_dir)

        # Initialize pipeline components
        self.loader = EvaluationReportLoader()
        self.extractor = SuggestionExtractor()
        self.style_analyzer = StyleAnalyzer()
        self.planner = ChangePlanner()
        self.renderer = DocumentRenderer()
        self.output = OutputManager(base_outputs_dir=self.config.output_dir)
        self.llm_gen = LLMContentGenerator(llm)
        self.llm_cleaner = LLMCleaner(llm)

    def print_step(self, step_name: str | None = None, step_output: str | None = None):
        """Report progress via callback."""
        if self.step_callback is None:
            return
        self.step_callback(step_name=step_name, step_output=step_output)

    def prepare_repo(self, repo_url_or_path: str):
        """Set the repository URL or path for processing."""
        self.repo_url_or_path = repo_url_or_path

    # =========================================================================
    # Main Pipeline
    # =========================================================================

    def run(
        self,
        report_path: str,
        repo_path: str | None = None,
        target_files: List[str] | None = None,
        max_files: int | None = None,
    ) -> str:
        """
        Run the documentation generation pipeline.

        Args:
            report_path: Path to the evaluation report JSON
            repo_path: Path to the repository (optional)
            target_files: Optional list of file paths to limit processing to
            max_files: Optional hard limit on number of files to process

        Returns:
            Path to the output directory
        """
        self.start_time = time.time()
        repo_path = repo_path or self.repo_url_or_path or ""

        # Override config with parameters if provided
        if target_files:
            self.config.target_files = target_files
        if max_files:
            self.config.max_files = max_files

        # Step 1: Load evaluation report
        report, report_abs = self._load_report(report_path)

        # Step 2: Read repository files
        target_file_list = self._build_target_file_list(report)
        files, missing = self._read_repo_files(repo_path, target_file_list)

        # Step 3: Filter files if needed
        files = self._filter_files(files, target_file_list)

        # Step 4: Analyze document style
        style = self._analyze_style(files)

        # Step 5: Extract suggestions
        suggestions = self._extract_suggestions(report)

        # Step 6: Plan changes
        plan = self._plan_changes(repo_path, style, suggestions, files)

        # Step 7: Group and filter edits
        edits_by_file = self._group_and_filter_edits(plan, target_file_list)

        # Step 8: Render documents
        revised, diff_stats = self._render_documents(
            edits_by_file, files, suggestions, plan, report
        )

        # Step 9: Write outputs
        out_dir = self._write_outputs(
            repo_path,
            report,
            files,
            revised,
            diff_stats,
            suggestions,
            plan,
            report_abs,
            missing,
        )

        self.print_step("Done", f"Generation completed! Output directory: {out_dir}")
        return out_dir

    # =========================================================================
    # Pipeline Step Methods
    # =========================================================================

    def _load_report(self, report_path: str) -> Tuple[EvaluationReport, str]:
        """Step 1: Load the evaluation report."""
        self.print_step(
            "LoadReport", f"Loading evaluation report from {report_path}..."
        )
        report, report_abs = self.loader.load(report_path)
        self.print_step("LoadReport", "Evaluation report loaded successfully")
        return report, report_abs

    def _build_target_file_list(self, report: EvaluationReport) -> List[str]:
        """Build list of target files from the evaluation report."""
        target_files: List[str] = []

        # README files
        if getattr(report, "readme_files", None):
            target_files.extend(report.readme_files)

        # Installation files
        if getattr(report, "installation_files", None):
            target_files.extend(report.installation_files)

        # Userguide files (from files or evaluation keys)
        userguide_files = self._extract_files_from_evaluation(
            report, "userguide_files", "userguide_evaluation"
        )
        target_files.extend(userguide_files)

        # Tutorial files (from files or evaluation keys)
        tutorial_files = self._extract_files_from_evaluation(
            report, "tutorial_files", "tutorial_evaluation"
        )
        target_files.extend(tutorial_files)

        # Submission requirements files
        if getattr(report, "submission_requirements_files", None):
            target_files.extend(report.submission_requirements_files)

        # Clean and deduplicate
        target_files = [p for p in target_files if isinstance(p, str) and p.strip()]
        target_files = list(dict.fromkeys(target_files))

        return target_files

    def _extract_files_from_evaluation(
        self,
        report: EvaluationReport,
        files_attr: str,
        eval_attr: str,
    ) -> List[str]:
        """Extract file paths from either files attribute or evaluation keys."""
        files: List[str] = []

        files_list = getattr(report, files_attr, None)
        if files_list:
            files.extend([p for p in files_list if isinstance(p, str)])
        elif getattr(report, eval_attr, None) and isinstance(
            getattr(report, eval_attr), dict
        ):
            for key in getattr(report, eval_attr).keys():
                if isinstance(key, str) and key.strip():
                    files.append(key)

        return files

    def _read_repo_files(
        self,
        repo_path: str,
        target_files: List[str],
    ) -> Tuple[Dict[str, str], List[str]]:
        """Step 2: Read files from the repository."""
        self.print_step(
            "ReadRepoFiles", f"Reading repository files from {repo_path}..."
        )
        reader = RepoReader(repo_path)

        if target_files:
            files, missing = reader.read_files(target_files)
        else:
            files, missing = reader.read_default_targets()

        self.print_step("ReadRepoFiles", f"Read {len(files)} files from repository")
        return files, missing

    def _filter_files(
        self,
        files: Dict[str, str],
        target_files: List[str],
    ) -> Dict[str, str]:
        """Step 3: Filter files to only include targets."""
        if not target_files:
            return files

        target_basenames = {os.path.basename(f) for f in target_files}
        target_paths = set(target_files)

        filtered = {}
        for fpath, content in files.items():
            if fpath in target_paths or os.path.basename(fpath) in target_basenames:
                filtered[fpath] = content

        if len(filtered) < len(files):
            self.print_step(
                "FilterFiles",
                f"Limiting to {len(filtered)} target files (from {len(files)})",
            )

        return filtered

    def _analyze_style(self, files: Dict[str, str]) -> StyleProfile:
        """Step 4: Analyze document style."""
        self.print_step("AnalyzeStyle", "Analyzing document style and formatting...")
        style = self.style_analyzer.analyze(files)
        self.print_step("AnalyzeStyle", "Document style analysis completed")
        return style

    def _extract_suggestions(self, report: EvaluationReport) -> List[SuggestionItem]:
        """Step 5: Extract suggestions from the report."""
        self.print_step(
            "ExtractSuggestions", "Extracting suggestions from evaluation report..."
        )
        suggestions = self.extractor.extract(report)
        self.print_step("Suggestions", f"Extracted {len(suggestions)} suggestions")
        return suggestions

    def _plan_changes(
        self,
        repo_path: str,
        style: StyleProfile,
        suggestions: List[SuggestionItem],
        files: Dict[str, str],
    ) -> DocumentPlan:
        """Step 6: Plan changes based on suggestions."""
        self.print_step("PlanChanges", "Planning changes based on suggestions...")
        plan = self.planner.build_plan(
            repo_path=repo_path,
            style=style,
            suggestions=suggestions,
            available_files=files,
        )
        file_count = len(set(e.file_path for e in plan.planned_edits))
        self.print_step(
            "PlannedEdits",
            f"Planned {len(plan.planned_edits)} edits across {file_count} files",
        )
        return plan

    def _group_and_filter_edits(
        self,
        plan: DocumentPlan,
        target_files: List[str],
    ) -> Dict[str, List[PlannedEdit]]:
        """Step 7: Group edits by file and filter by targets."""
        edits_by_file: Dict[str, List[PlannedEdit]] = {}
        for e in plan.planned_edits:
            edits_by_file.setdefault(e.file_path, []).append(e)

        # Filter by target files
        if target_files:
            edits_by_file = self._filter_edits_by_targets(edits_by_file, target_files)

        # Apply max_files limit
        if self.config.max_files and len(edits_by_file) > self.config.max_files:
            all_files = list(edits_by_file.keys())
            limited = all_files[: self.config.max_files]
            self.print_step(
                "HardLimit",
                f"Limited to {len(limited)} files (from {len(edits_by_file)})",
            )
            edits_by_file = {k: edits_by_file[k] for k in limited}

        return edits_by_file

    def _filter_edits_by_targets(
        self,
        edits_by_file: Dict[str, List[PlannedEdit]],
        target_files: List[str],
    ) -> Dict[str, List[PlannedEdit]]:
        """Filter edits to only include target files."""
        target_basenames = {os.path.basename(f) for f in target_files}
        target_paths = set(target_files)
        target_normalized = {os.path.normpath(f) for f in target_files}

        filtered = {}
        for fpath, edits in edits_by_file.items():
            fpath_norm = os.path.normpath(fpath)
            fpath_base = os.path.basename(fpath)

            if (
                fpath in target_paths
                or fpath_norm in target_normalized
                or fpath_base in target_basenames
            ):
                filtered[fpath] = edits

        skipped = len(edits_by_file) - len(filtered)
        self.print_step(
            "FilterEdits",
            f"Matched {len(filtered)} of {len(edits_by_file)} files (skipping {skipped})",
        )

        return filtered

    def _render_documents(
        self,
        edits_by_file: Dict[str, List[PlannedEdit]],
        files: Dict[str, str],
        suggestions: List[SuggestionItem],
        plan: DocumentPlan,
        report: EvaluationReport,
    ) -> Tuple[Dict[str, str], Dict[str, dict]]:
        """Step 8: Render documents with LLM."""
        self.print_step(
            "RenderDocuments",
            f"Rendering documents with LLM (processing {len(edits_by_file)} files)...",
        )

        revised: Dict[str, str] = {}
        diff_stats: Dict[str, dict] = {}
        total_files = len(edits_by_file)

        for idx, (fpath, edits) in enumerate(edits_by_file.items(), 1):
            self.print_step(
                "ProcessingFile",
                f"Processing {fpath} ({idx}/{total_files}) - {len(edits)} edits",
            )

            original_content = files.get(fpath, "")
            content, stats = self._process_file_edits(
                fpath, edits, original_content, suggestions, plan
            )

            # Clean markdown files
            content = self._clean_content(fpath, content)

            revised[fpath] = content
            diff_stats[fpath] = stats

            self.print_step(
                "RenderedFile",
                f"Completed {fpath} - added {stats.get('added_lines', 0)} lines",
            )

        return revised, diff_stats

    def _process_file_edits(
        self,
        fpath: str,
        edits: List[PlannedEdit],
        original_content: str,
        suggestions: List[SuggestionItem],
        plan: DocumentPlan,
    ) -> Tuple[str, dict]:
        """Process edits for a single file."""
        # Group suggestions by type
        file_suggestions = []
        full_replace_edits = []
        section_edits = []

        for e in edits:
            suggestion = self._find_suggestion(e.suggestion_id, suggestions)
            if suggestion:
                file_suggestions.append(suggestion)
                if e.edit_type == "full_replace":
                    full_replace_edits.append(e)
                else:
                    section_edits.append(e)

        # Write debug info if enabled
        if self.config.debug_output:
            self._write_debug_info(
                fpath, edits, file_suggestions, full_replace_edits, section_edits
            )

        content = original_content
        total_stats = {"added_lines": 0}

        # Generate content
        if full_replace_edits:
            content, stats = self._generate_full_document(
                fpath, full_replace_edits, file_suggestions, original_content
            )
            total_stats["added_lines"] += stats.get("added_lines", 0)
        else:
            # Handle section edits
            for e in section_edits:
                suggestion = self._find_suggestion(e.suggestion_id, suggestions)
                if suggestion and (
                    not e.content_template or not e.content_template.strip()
                ):
                    gen_section = self._generate_section(
                        e, suggestion, plan.style_profile, original_content
                    )
                    if gen_section:
                        e.content_template = gen_section

                content, stats = self.renderer.apply_edit(content, e)
                total_stats["added_lines"] += stats.get("added_lines", 0)

        # Apply remaining non-full_replace edits
        for e in edits:
            if e.edit_type != "full_replace":
                content, stats = self.renderer.apply_edit(content, e)
                total_stats["added_lines"] += stats.get("added_lines", 0)

        return content, total_stats

    def _find_suggestion(
        self,
        suggestion_id: Optional[str],
        suggestions: List[SuggestionItem],
    ) -> Optional[SuggestionItem]:
        """Find a suggestion by ID."""
        if not suggestion_id:
            return None
        return next((s for s in suggestions if s.id == suggestion_id), None)

    def _generate_full_document(
        self,
        fpath: str,
        full_replace_edits: List[PlannedEdit],
        file_suggestions: List[SuggestionItem],
        original_content: str,
    ) -> Tuple[str, dict]:
        """Generate a full document using LLM."""
        self.print_step(
            "GeneratingContent",
            f"Generating full document for {fpath} with {len(file_suggestions)} suggestions (SINGLE CALL)...",
        )

        # Merge suggestions into evaluation report
        merged_report = self._build_merged_evaluation_report(file_suggestions)

        gen_content, gen_usage = self.llm_gen.generate_full_document(
            target_file=fpath,
            evaluation_report=merged_report,
            context=original_content,
            original_content=original_content,
        )

        if isinstance(gen_content, str) and gen_content:
            self.print_step(
                "LLMFullDoc",
                f"Generated full document for {fpath} ({gen_usage.get('total_tokens', 0)} tokens)",
            )
            # Apply content to all full_replace edits
            for e in full_replace_edits:
                e.content_template = gen_content

            added_lines = len(gen_content.splitlines()) - len(
                original_content.splitlines()
            )
            return gen_content, {"added_lines": max(0, added_lines)}

        return original_content, {"added_lines": 0}

    def _build_merged_evaluation_report(
        self,
        suggestions: List[SuggestionItem],
    ) -> Dict[str, Any]:
        """Build merged evaluation report from suggestions."""
        suggestions_list = []
        for idx, s in enumerate(suggestions, 1):
            suggestions_list.append(
                {
                    "suggestion_number": idx,
                    "category": s.category if hasattr(s, "category") else "general",
                    "content_guidance": s.content_guidance,
                }
            )

        return {
            "total_suggestions": len(suggestions),
            "integration_instruction": (
                f"Integrate ALL {len(suggestions)} suggestions below into ONE cohesive document. "
                f"Do NOT create {len(suggestions)} separate versions."
            ),
            "suggestions": suggestions_list,
        }

    def _generate_section(
        self,
        edit: PlannedEdit,
        suggestion: SuggestionItem,
        style: StyleProfile,
        context: str,
    ) -> Optional[str]:
        """Generate a section using LLM."""
        self.print_step(
            "GeneratingContent",
            f"Generating section for {edit.suggestion_id} using LLM...",
        )

        gen_section, gen_usage = self.llm_gen.generate_section(
            suggestion=suggestion,
            style=style,
            context=context,
        )

        if isinstance(gen_section, str) and gen_section:
            self.print_step(
                "LLMSection",
                f"Generated section for {edit.suggestion_id} ({gen_usage.get('total_tokens', 0)} tokens)",
            )

            # Ensure header present
            if gen_section.lstrip().startswith("#"):
                return gen_section
            else:
                title = edit.anchor.get("value", "").strip() or ""
                return f"## {title}\n\n{gen_section}" if title else gen_section

        return None

    def _clean_content(self, fpath: str, content: str) -> str:
        """Clean formatting for markdown files."""
        if not content:
            return content

        if not fpath.endswith((".md", ".rst", ".Rmd", ".Rd")):
            return content

        if not self.config.clean_output:
            return content

        try:
            self.print_step("CleaningContent", f"Cleaning formatting for {fpath}...")
            cleaned, _usage = self.llm_cleaner.clean_readme(content)
            if isinstance(cleaned, str) and cleaned.strip():
                return cleaned
        except Exception as e:
            logger.warning(f"Failed to clean content for {fpath}: {e}")

        return content

    def _write_outputs(
        self,
        repo_path: str,
        report: EvaluationReport,
        files: Dict[str, str],
        revised: Dict[str, str],
        diff_stats: Dict[str, dict],
        suggestions: List[SuggestionItem],
        plan: DocumentPlan,
        report_abs: str,
        missing: List[str],
    ) -> str:
        """Step 9: Write all outputs."""
        # Determine output key
        out_repo_key = self._get_output_repo_key(repo_path, report)

        self.print_step("WriteOutputs", f"Writing outputs to {out_repo_key}...")
        out_dir = self.output.prepare_output_dir(out_repo_key)

        # Merge all files
        all_files: Dict[str, str] = dict(files)
        all_files.update(revised)

        # Add original copies for comparison
        if self.config.write_originals:
            for orig_path, orig_content in files.items():
                all_files[self._original_copy_name(orig_path)] = orig_content

        self.print_step(
            "WritingFiles", f"Writing {len(all_files)} files to output directory..."
        )
        artifacts = self.output.write_files(
            out_dir, all_files, diff_stats_by_file=diff_stats
        )

        # Write manifest
        manifest = GenerationManifest(
            repo_url=report.repo_url,
            report_path=report_abs,
            output_dir=out_dir,
            suggestions=suggestions,
            planned_edits=plan.planned_edits,
            artifacts=artifacts,
            skipped=missing,
        )
        self.print_step("WritingManifest", "Writing generation manifest...")
        self.output.write_manifest(out_dir, manifest)

        # Write generation report
        self.print_step("WritingReport", "Writing generation report...")
        self._write_generation_report(
            out_dir,
            report.repo_url or str(self.repo_url_or_path or ""),
            plan,
            diff_stats,
            suggestions,
            artifacts,
            missing,
        )

        return out_dir

    def _get_output_repo_key(self, repo_path: str, report: EvaluationReport) -> str:
        """Determine the output directory key."""
        if repo_path and os.path.isdir(repo_path):
            return os.path.basename(os.path.normpath(repo_path))
        elif report.repo_url:
            try:
                author, name = parse_repo_url(report.repo_url)
                return f"{author}_{name}"
            except Exception:
                return report.repo_url
        else:
            return self.repo_url_or_path or "repo"

    def _original_copy_name(self, path: str) -> str:
        """Generate name for original file backup."""
        if "." in path:
            base, ext = path.rsplit(".", 1)
            return f"{base}.original.{ext}"
        return f"{path}.original"

    # =========================================================================
    # Debug Output
    # =========================================================================

    def _write_debug_info(
        self,
        fpath: str,
        edits: List[PlannedEdit],
        file_suggestions: List[SuggestionItem],
        full_replace_edits: List[PlannedEdit],
        section_edits: List[PlannedEdit],
    ):
        """Write debug information to files."""
        debug_dir = self.config.debug_dir
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
                    "content_guidance": (
                        s.content_guidance[:200] + "..."
                        if len(s.content_guidance or "") > 200
                        else s.content_guidance
                    ),
                    "target_files": s.target_files,
                }
                for s in file_suggestions
            ],
            "timestamp": datetime.datetime.now().isoformat(),
        }

        grouping_file = os.path.join(debug_dir, f"{safe_filename}_grouping.json")
        with open(grouping_file, "w", encoding="utf-8") as f:
            json.dump(grouping_info, f, indent=2, ensure_ascii=False)

    # =========================================================================
    # Report Generation
    # =========================================================================

    def _get_generation_time(self) -> str:
        """Get formatted generation time with start, end, and duration."""
        if self.start_time is None:
            return "Not tracked"

        end_time = time.time()
        duration = end_time - self.start_time

        start_str = datetime.datetime.fromtimestamp(self.start_time).strftime(
            "%H:%M:%S"
        )
        end_str = datetime.datetime.fromtimestamp(end_time).strftime("%H:%M:%S")

        if duration < 60:
            duration_str = f"{duration:.1f}s"
        elif duration < 3600:
            duration_str = f"{duration / 60:.1f}m"
        else:
            duration_str = f"{duration / 3600:.1f}h"

        return f"{start_str} -> {end_str} ({duration_str})"

    def _write_generation_report(
        self,
        out_dir: str,
        repo_url: str,
        plan: DocumentPlan,
        diff_stats: Dict[str, dict],
        suggestions: List[SuggestionItem],
        artifacts,
        skipped: List[str],
    ):
        """Write human-readable generation report."""
        report_generator = GenerationReportWriter(
            out_dir=out_dir,
            repo_url=repo_url,
            plan=plan,
            diff_stats=diff_stats,
            suggestions=suggestions,
            artifacts=artifacts,
            skipped=skipped,
            generation_time=self._get_generation_time(),
        )
        report_generator.write()


class GenerationReportWriter:
    """
    Writes the human-readable generation report.

    Extracted from DocumentationGenerationManager to separate concerns.
    """

    def __init__(
        self,
        out_dir: str,
        repo_url: str,
        plan: DocumentPlan,
        diff_stats: Dict[str, dict],
        suggestions: List[SuggestionItem],
        artifacts,
        skipped: List[str],
        generation_time: str,
    ):
        self.out_dir = out_dir
        self.repo_url = repo_url
        self.plan = plan
        self.diff_stats = diff_stats
        self.suggestions = suggestions
        self.artifacts = artifacts
        self.skipped = skipped
        self.generation_time = generation_time

    def write(self) -> str:
        """Write the report and return the path."""
        lines = self._build_report()
        report_md = "\n".join(lines)

        dest = os.path.join(self.out_dir, "GENERATION_REPORT.md")
        with open(dest, "w", encoding="utf-8") as f:
            f.write(report_md)

        return dest

    def _build_report(self) -> List[str]:
        """Build the report content."""
        lines = []

        # Header
        lines.extend(self._build_header())

        # Summary
        lines.extend(self._build_summary())

        # Key metrics
        lines.extend(self._build_metrics())

        # Files improved
        lines.extend(self._build_files_section())

        # Skipped files note
        if self.skipped:
            lines.extend(self._build_skipped_section())

        return lines

    def _build_header(self) -> List[str]:
        """Build report header."""
        time_parts = self.generation_time.split(" -> ")
        start_time = time_parts[0] if len(time_parts) > 0 else "Not tracked"
        end_duration = time_parts[1] if len(time_parts) > 1 else "Not tracked"

        if " (" in end_duration:
            end_time, duration = end_duration.split(" (")
            duration = duration.rstrip(")")
        else:
            end_time = end_duration
            duration = "Not tracked"

        return [
            "# Documentation Generation Report\n",
            f"**Repository:** {self.repo_url}\n",
            f"**Generated:** {self.out_dir}\n",
            f"**Processing Timeline:**\n",
            f"- **Start Time:** {start_time}\n",
            f"- **End Time:** {end_time}\n",
            f"- **Duration:** {duration}\n",
        ]

    def _build_summary(self) -> List[str]:
        """Build summary section."""
        return [
            "\n## Summary\n",
            "This is a report of automated documentation enhancements generated by BioGuider.\n",
            "\nOur AI analyzed your existing documentation to identify areas for improvement "
            "based on standards for high-quality scientific software. It then automatically "
            "rewrote the files to be more accessible and useful for biomedical researchers.\n",
            "\nThis changelog provides a transparent record of what was modified and why. "
            "We encourage you to review the changes before committing. Original file versions "
            "are backed up with a `.original` extension.\n",
        ]

    def _build_metrics(self) -> List[str]:
        """Build key metrics section."""
        lines = []

        # Calculate statistics
        total_improvements = len(self.plan.planned_edits)
        file_stats = {}
        score_stats = {"Excellent": 0, "Good": 0, "Fair": 0, "Poor": 0}
        processed_ids = set()

        for e in self.plan.planned_edits:
            file_stats[e.file_path] = file_stats.get(e.file_path, 0) + 1

            sug = self._find_suggestion(e.suggestion_id)
            if sug and sug.source and e.suggestion_id not in processed_ids:
                score = sug.source.get("score", "")
                if score in score_stats:
                    score_stats[score] += 1
                processed_ids.add(e.suggestion_id)

        total_lines_added = sum(
            s.get("added_lines", 0) for s in self.diff_stats.values()
        )

        # Count processed suggestions
        processed_count = sum(
            1
            for s in self.suggestions
            if s.source and s.source.get("score", "") in ("Fair", "Poor")
        )
        fixed_count = len(
            [
                sid
                for sid in processed_ids
                if self._find_suggestion(sid)
                and self._find_suggestion(sid).source
                and self._find_suggestion(sid).source.get("score", "")
                in ("Fair", "Poor")
            ]
        )

        success_rate = (
            (fixed_count / processed_count * 100) if processed_count > 0 else 0
        )

        lines.extend(
            [
                "\n### Key Metrics\n",
                f"- **Success Rate:** {success_rate:.1f}% ({fixed_count} of {processed_count} processed suggestions addressed)\n",
                f"- **Total Impact:** {total_improvements} improvements across {len(file_stats)} files\n",
                f"- **Content Added:** {total_lines_added} lines of enhanced documentation\n",
            ]
        )

        # Priority breakdown
        lines.append("\n### Priority Breakdown\n")
        for score in ["Poor", "Fair"]:
            if score_stats[score] > 0:
                lines.append(
                    f"- **{score} Priority:** {score_stats[score]} items -> 100% addressed\n"
                )

        return lines

    def _build_files_section(self) -> List[str]:
        """Build files improved section."""
        lines = ["\n## Files Improved\n"]

        by_file = {}
        for e in self.plan.planned_edits:
            by_file.setdefault(e.file_path, []).append(e)

        for file_path, edits in by_file.items():
            added_lines = self.diff_stats.get(file_path, {}).get("added_lines", 0)
            lines.append(f"\n### {file_path}\n")
            lines.append(
                f"**Changes made:** {len(edits)} improvement(s), {added_lines} lines added\n"
            )

            for e in edits:
                sug = self._find_suggestion(e.suggestion_id)
                action_desc = self._get_action_description(e, sug)
                lines.append(f"- **{action_desc}**")

                # Show evaluation reasoning
                if sug and sug.source:
                    score = sug.source.get("score", "")
                    category = sug.category or ""
                    cat_display = (
                        category.split(".")[-1].replace("_", " ").title()
                        if category
                        else ""
                    )

                    if score and cat_display:
                        lines.append(f"  - *Reason:* [{cat_display} - {score}]")
                    elif score:
                        lines.append(f"  - *Reason:* [{score}]")

                lines.append("")

        return lines

    def _build_skipped_section(self) -> List[str]:
        """Build skipped files section."""
        lines = [
            "\n## Note\n",
            "The following files were not modified as they were not found in the repository:",
        ]
        for rel in self.skipped:
            lines.append(f"- {rel}")
        return lines

    def _find_suggestion(
        self, suggestion_id: Optional[str]
    ) -> Optional[SuggestionItem]:
        """Find suggestion by ID."""
        if not suggestion_id:
            return None
        return next((s for s in self.suggestions if s.id == suggestion_id), None)

    def _get_action_description(
        self, edit: PlannedEdit, sug: Optional[SuggestionItem]
    ) -> str:
        """Get human-readable action description."""
        action_key = sug.action if sug else edit.edit_type
        section = edit.anchor.get("value", "General improvements")

        if action_key == "full_replace" and sug:
            category = sug.category or ""
            cat_lower = category.lower()

            if "readme" in cat_lower:
                return "Enhanced README documentation"
            elif "tutorial" in cat_lower:
                return "Improved tutorial content"
            elif "userguide" in cat_lower:
                return "Enhanced user guide documentation"
            elif "installation" in cat_lower:
                return "Improved installation instructions"
            elif "dependencies" in cat_lower:
                return "Enhanced dependency information"
            elif "readability" in cat_lower:
                return "Improved readability and clarity"
            else:
                cat_display = category.split(".")[-1].replace("_", " ").title()
                return (
                    f"Enhanced {cat_display}"
                    if cat_display
                    else "Comprehensive rewrite"
                )

        action_map = {
            "append_section": f'Added "{section}" section',
            "insert_after_header": f'Enhanced content in "{section}"',
            "rmarkdown_integration": f'Integrated improvements in "{section}"',
            "replace_intro_block": f'Improved "{section}" section',
            "add_dependencies_section": "Added dependencies information",
            "add_system_requirements_section": "Added system requirements",
            "add_hardware_requirements": "Added hardware requirements",
        }

        return action_map.get(action_key, f"Improved {action_key}")
