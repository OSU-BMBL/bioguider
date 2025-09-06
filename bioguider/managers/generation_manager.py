from __future__ import annotations

import os
from pathlib import Path
from typing import Tuple, Dict

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
from bioguider.generation.models import GenerationManifest
from bioguider.utils.file_utils import parse_repo_url


class DocumentationGenerationManager:
    def __init__(self, llm, step_callback):
        self.llm = llm
        self.step_callback = step_callback
        self.repo_url_or_path: str | None = None

        self.loader = EvaluationReportLoader()
        self.extractor = SuggestionExtractor()
        self.style_analyzer = StyleAnalyzer()
        self.planner = ChangePlanner()
        self.renderer = DocumentRenderer()
        self.output = OutputManager()
        self.llm_gen = LLMContentGenerator(llm)
        self.llm_cleaner = LLMCleaner(llm)

    def print_step(self, step_name: str | None = None, step_output: str | None = None):
        if self.step_callback is None:
            return
        self.step_callback(step_name=step_name, step_output=step_output)

    def prepare_repo(self, repo_url_or_path: str):
        self.repo_url_or_path = repo_url_or_path

    def run(self, report_path: str, repo_path: str | None = None) -> str:
        repo_path = repo_path or self.repo_url_or_path or ""
        self.print_step(step_name="LoadReport", step_output=f"report_path={report_path}")
        report, report_abs = self.loader.load(report_path)

        self.print_step(step_name="ReadRepoFiles", step_output=f"repo_path={repo_path}")
        reader = RepoReader(repo_path)
        # Prefer report-listed files if available
        target_files = []
        if report.readme_files:
            target_files.extend(report.readme_files)
        if report.installation_files:
            target_files.extend(report.installation_files)
        target_files = list(dict.fromkeys(target_files))  # de-dup
        files, missing = reader.read_files(target_files) if target_files else reader.read_default_targets()

        self.print_step(step_name="AnalyzeStyle", step_output=f"files={[p for p in files.keys()]}")
        style = self.style_analyzer.analyze(files)

        self.print_step(step_name="ExtractSuggestions")
        suggestions = self.extractor.extract(report)
        self.print_step(step_name="Suggestions", step_output=f"count={len(suggestions)} ids={[s.id for s in suggestions]}")

        self.print_step(step_name="PlanChanges")
        plan = self.planner.build_plan(repo_path=repo_path, style=style, suggestions=suggestions, available_files=files)
        self.print_step(step_name="PlannedEdits", step_output=f"count={len(plan.planned_edits)} files={list(set(e.file_path for e in plan.planned_edits))}")

        self.print_step(step_name="RenderDocuments")
        # Apply edits cumulatively per file to ensure multiple suggestions are realized
        revised: Dict[str, str] = {}
        diff_stats: Dict[str, dict] = {}
        edits_by_file: Dict[str, list] = {}
        for e in plan.planned_edits:
            edits_by_file.setdefault(e.file_path, []).append(e)
        for fpath, edits in edits_by_file.items():
            content = files.get(fpath, "")
            total_stats = {"added_lines": 0}
            for e in edits:
                # Generate LLM content for section if template is generic
                context = files.get(fpath, "")
                gen_section, gen_usage = self.llm_gen.generate_section(
                    suggestion=next((s for s in suggestions if s.id == e.suggestion_id), None) if e.suggestion_id else None,
                    style=plan.style_profile,
                    context=context,
                ) if e.suggestion_id else ""
                if isinstance(gen_section, str) and gen_section:
                    self.print_step(step_name="LLMSection", step_output=f"file={fpath} suggestion={e.suggestion_id} tokens={gen_usage.get('total_tokens', 0)}\n{gen_section}")
                    # Ensure header present
                    if gen_section.lstrip().startswith("#"):
                        e.content_template = gen_section
                    else:
                        title = e.anchor.get('value', '').strip() or ''
                        e.content_template = f"## {title}\n\n{gen_section}" if title else gen_section
                content, stats = self.renderer.apply_edit(content, e)
                total_stats["added_lines"] = total_stats.get("added_lines", 0) + stats.get("added_lines", 0)
            revised[fpath] = content
            diff_stats[fpath] = total_stats
            self.print_step(step_name="RenderedFile", step_output=f"file={fpath} added_lines={total_stats['added_lines']}")

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

        self.print_step(step_name="WriteOutputs", step_output=f"repo_key={out_repo_key}")
        out_dir = self.output.prepare_output_dir(out_repo_key)
        artifacts = self.output.write_files(out_dir, revised)

        manifest = GenerationManifest(
            repo_url=report.repo_url,
            report_path=report_abs,
            output_dir=out_dir,
            suggestions=suggestions,
            planned_edits=plan.planned_edits,
            artifacts=artifacts,
            skipped=missing,
        )
        self.output.write_manifest(out_dir, manifest)
        self.print_step(step_name="Done", step_output=f"output_dir={out_dir}")
        return out_dir


