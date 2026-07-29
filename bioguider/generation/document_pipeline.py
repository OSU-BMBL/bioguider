import json
import os
from pathlib import Path

from langchain_openai.chat_models.base import BaseChatOpenAI

from bioguider.utils.constants import (
    EvaluationTypeEnum,
    PrimaryLanguageEnum,
    ProjectMetadata,
    ProjectTypeEnum,
)
from bioguider.generation.llm_content_generator import LLMContentGenerator
from bioguider.generation.markdown_polisher import (
    MarkdownPolisher,
    _accept_polish_if_safe,
)


class DocumentPipeline:
    """
    BioGuider evaluation + generation pipeline for a single repository.

    Usage::

        pipeline = DocumentPipeline(repo_path)
        pipeline.prepare_repo(llm)          # once — builds CodeStructureDb + FAISS

        # per benchmark cell (different llm per model):
        report, refined = pipeline.evaluate_and_refine_document(
            llm=model_llm,
            doc_repo_path=output_dir,
            doc_path="vignette.level_5.corrupted.Rmd",
            eval_type=EvaluationTypeEnum.TUTORIAL,
        )

    ``prepare_repo`` is LLM-independent (pure AST + embedding index).
    ``evaluate_and_refine_document`` accepts a per-call LLM so different
    models can be benchmarked against the same pre-built databases.
    """

    def __init__(self, repo_path: str) -> None:
        self.repo_path = repo_path
        self.code_structure_db = None
        self.summary_file_db = None

    def prepare_repo(self, llm: BaseChatOpenAI) -> "DocumentPipeline":
        """
        Build CodeStructureDb and SummarizedFilesDb from repo_path.

        Must be called once before evaluate_and_refine_document.
        Returns self so calls can be chained.

        Args:
            llm: LLM passed to EvaluationManager (used for its internal init;
                 the actual DB build is LLM-independent).
        """
        from bioguider.managers.evaluation_manager import EvaluationManager

        manager = EvaluationManager(llm, step_callback=None)
        manager.prepare_repo(self.repo_path)
        self.code_structure_db = manager.code_structure_db
        self.summary_file_db = manager.summary_file_db
        return self

    def evaluate_and_refine_document(
        self,
        llm: BaseChatOpenAI,
        doc_repo_path: str,
        doc_path: str,
        eval_type: EvaluationTypeEnum,
        report_output_path: str | None = None,
        eval_report_output_path: str | None = None,
        suggestion_categories: list[str] | None = None,
        polish: bool = True,
    ) -> tuple[dict, str]:
        """
        Evaluate a document then generate a refined version.

        Args:
            llm: Chat LLM used for both the evaluation task and the generator.
            doc_repo_path: Directory that contains the document on disk.
            doc_path: Filename relative to doc_repo_path.
            eval_type: Which BioGuider evaluation task to run.
            report_output_path: If given, write the merged generator-input
                report (flat list of suggestions, what the generator actually
                consumes) to this path as JSON.
            eval_report_output_path: If given, write the raw evaluation
                results (per-dimension scores + assessment text + structured
                findings, before they are flattened into the merged report)
                to this path as JSON.  Useful for inspecting what the
                evaluation task actually found.
            suggestion_categories: If given, only suggestions whose ``category``
                is in this list are passed to the generator.  Use
                ``["readability_errors", "readability"]`` to restrict the
                pipeline to direct error-fixes only (no setup / reproducibility
                / structure improvements).  ``None`` (default) passes every
                category, which is the original behaviour.
            polish: If True (default), run a narrow surface-markdown polish
                pass over the generated document — fixes residual broken
                inline-code spans / image / link syntax / prose typos that
                the evaluation tasks do not emit explicit findings for.
                Gated by a structural guardrail (length, fence count, and
                header count must stay within tolerance of the pre-polish
                output); if the polish drifts, the unpolished generator
                output is returned.  Set to False for ablation runs that
                want to measure the generator in isolation.

        Returns:
            (merged_report, refined_content)
            merged_report is the dict passed to the generation step;
            refined_content is the LLM-generated improved document.
        """
        meta = ProjectMetadata(
            url=doc_repo_path,
            project_type=ProjectTypeEnum.unknown,
            primary_language=PrimaryLanguageEnum.unknown,
            repo_name=Path(doc_repo_path).name,
        )
        gitignore_path = str(Path(doc_repo_path, ".gitignore"))

        eval_results, _ = _run_evaluation(
            llm=llm,
            repo_path=doc_repo_path,
            gitignore_path=gitignore_path,
            doc_path=doc_path,
            meta=meta,
            eval_type=eval_type,
            code_structure_db=self.code_structure_db,
            summary_file_db=self.summary_file_db,
        )

        merged_report = _build_merged_report(
            eval_results, doc_path, eval_type,
            suggestion_categories=suggestion_categories,
        )

        if report_output_path:
            os.makedirs(os.path.dirname(report_output_path), exist_ok=True)
            with open(report_output_path, "w", encoding="utf-8") as _f:
                json.dump(merged_report, _f, indent=2, default=str)

        if eval_report_output_path:
            os.makedirs(os.path.dirname(eval_report_output_path), exist_ok=True)
            with open(eval_report_output_path, "w", encoding="utf-8") as _f:
                json.dump(
                    _serialise_eval_results(eval_results),
                    _f,
                    indent=2,
                    default=str,
                )

        original_content = Path(doc_repo_path, doc_path).read_text(
            encoding="utf-8", errors="replace"
        )

        generator = LLMContentGenerator(llm)
        refined, _ = generator.generate_full_document(
            target_file=doc_path,
            evaluation_report=merged_report,
            context=original_content,
            original_content=original_content,
        )

        # Surface-markdown polish pass.  Closes the inline_code / image /
        # link / typo gap where ``simple`` historically beats ``pipeline``
        # because the evaluation tasks emit no targeted findings for those
        # categories.  Constructed with the same ``llm`` so the polish is
        # attributed to the model under test — never a hard-coded default.
        # ``_accept_polish_if_safe`` guarantees we never regress structure
        # relative to ``refined``: if the polish drifts on length, fence
        # count, or header count, we keep the pre-polish output.
        if polish and refined:
            polisher = MarkdownPolisher(llm)
            polished, _ = polisher.polish(refined)
            refined = _accept_polish_if_safe(refined, polished, original_content)

        return merged_report, refined or original_content


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _append_consistency_suggestions(
    suggestions: list,
    next_idx: int,
    consistency_eval,
    keep_categories: set | None,
) -> int:
    """Flatten ``consistency_evaluation.development`` into ``suggestions``.

    Each entry in the consistency task's ``development`` list (the concrete
    inconsistency findings — undefined CLI flags, mismatched names,
    contradicted docstrings, etc.) becomes one suggestion with
    ``category="consistency"`` so the generator can act on them.  The
    high-level ``assessment`` text is intentionally not emitted — it is
    summary commentary, not an actionable line.

    Returns the next suggestion index to use.
    """
    if consistency_eval is None:
        return next_idx
    if keep_categories is not None and "consistency" not in keep_categories:
        return next_idx
    items = getattr(consistency_eval, "development", None) or []
    for item in items:
        if not item:
            continue
        suggestions.append({
            "suggestion_number": next_idx,
            "category": "consistency",
            "content_guidance": str(item),
        })
        next_idx += 1
    return next_idx


def _serialise_eval_results(eval_results: dict | None) -> dict:
    """Convert ``eval_results`` (``{doc_path: PydanticResult}``) into a
    JSON-friendly dict.

    Pydantic v2 result models are dumped via ``model_dump()``; anything
    else is passed through unchanged and relies on ``json.dump``'s
    ``default=str`` fallback at the call site.
    """
    out: dict = {}
    for k, v in (eval_results or {}).items():
        if hasattr(v, "model_dump"):
            try:
                out[k] = v.model_dump()
                continue
            except Exception:
                pass
        out[k] = v
    return out


def _run_evaluation(
    llm,
    repo_path: str,
    gitignore_path: str,
    doc_path: str,
    meta: ProjectMetadata,
    eval_type: EvaluationTypeEnum,
    code_structure_db=None,
    summary_file_db=None,
):
    """Instantiate the correct evaluation task and run it on a single file."""
    kwargs = dict(
        llm=llm,
        repo_path=repo_path,
        gitignore_path=gitignore_path,
        meta_data=meta,
        step_callback=None,
        summarized_files_db=summary_file_db,
        collected_files=[doc_path],
    )
    if eval_type == EvaluationTypeEnum.TUTORIAL:
        from bioguider.agents.evaluation_tutorial_task import EvaluationTutorialTask
        task = EvaluationTutorialTask(**kwargs, code_structure_db=code_structure_db)
    elif eval_type == EvaluationTypeEnum.README:
        from bioguider.agents.evaluation_readme_task import EvaluationREADMETask
        task = EvaluationREADMETask(**kwargs)
    elif eval_type == EvaluationTypeEnum.INSTALLATION:
        from bioguider.agents.evaluation_installation_task import EvaluationInstallationTask
        task = EvaluationInstallationTask(**kwargs)
    elif eval_type == EvaluationTypeEnum.USERGUIDE:
        from bioguider.agents.evaluation_userguide_task import EvaluationUserGuideTask
        task = EvaluationUserGuideTask(**kwargs, code_structure_db=code_structure_db)
    else:
        raise ValueError(f"Unsupported eval_type: {eval_type}")

    return task.evaluate()


def _build_merged_report(
    eval_results: dict,
    doc_path: str,
    eval_type: EvaluationTypeEnum,
    suggestion_categories: list[str] | None = None,
) -> dict:
    """Convert a raw evaluation result dict into the merged-report format
    that LLMContentGenerator.generate_full_document expects.

    Args:
        suggestion_categories: If given, only suggestions whose category is in
            this set are included.  ``None`` (default) keeps every category.
    """
    _keep = set(suggestion_categories) if suggestion_categories is not None else None

    suggestions = []
    idx = 1

    if eval_type == EvaluationTypeEnum.TUTORIAL:
        result = eval_results.get(doc_path) if eval_results else None
        if result is not None:
            # Consistency findings (CLI flag inconsistencies, mismatched
            # function names, prose-vs-code contradictions) are the
            # highest-value, repo-aware output of the evaluation pipeline
            # — and they're short and dense.  Emit them FIRST so they
            # survive the generator's prompt-truncation budget even when
            # the long-tail readability findings would otherwise push
            # them past the cutoff.
            idx = _append_consistency_suggestions(
                suggestions, idx, getattr(result, "consistency_evaluation", None), _keep,
            )
            te = result.tutorial_evaluation
            if te is not None:
                category_fields = [
                    ("readability_errors", te.readability_errors_found or []),
                    ("readability", te.readability_suggestions or []),
                    ("setup", te.setup_and_dependencies_suggestions or []),
                    ("reproducibility", te.reproducibility_suggestions or []),
                    ("structure", te.structure_and_navigation_suggestions or []),
                    ("code_quality", te.executable_code_quality_suggestions or []),
                    ("result_verification", te.result_verification_suggestions or []),
                    ("performance", te.performance_and_resource_notes_suggestions or []),
                ]
                for category, items in category_fields:
                    if _keep is not None and category not in _keep:
                        continue
                    for item in items:
                        suggestions.append({
                            "suggestion_number": idx,
                            "category": category,
                            "content_guidance": item,
                        })
                        idx += 1

    elif eval_type == EvaluationTypeEnum.USERGUIDE:
        result = eval_results.get(doc_path) if eval_results else None
        if result is not None:
            # Consistency-first ordering, same rationale as TUTORIAL: keep
            # the repo-aware CLI / code / docstring findings at the front
            # of the merged report so they survive prompt truncation.
            idx = _append_consistency_suggestions(
                suggestions, idx, getattr(result, "consistency_evaluation", None), _keep,
            )
            ug = result.user_guide_evaluation
            if ug is not None:
                category_fields = [
                    ("readability_errors", ug.readability_errors_found or []),
                    ("readability", ug.readability_suggestions or []),
                    ("context_and_purpose", ug.context_and_purpose_suggestions or []),
                    ("error_handling", ug.error_handling_suggestions or []),
                ]
                for category, items in category_fields:
                    if _keep is not None and category not in _keep:
                        continue
                    for item in items:
                        suggestions.append({
                            "suggestion_number": idx,
                            "category": category,
                            "content_guidance": item,
                        })
                        idx += 1

    elif eval_type == EvaluationTypeEnum.README:
        for _file, result in (eval_results or {}).items():
            if result is None:
                continue
            se = getattr(result, "structured_evaluation", None)
            fe = getattr(result, "free_evaluation", None)
            for attr in [
                "readability_suggestions",
                "project_purpose_suggestions",
                "hardware_and_software_spec_suggestions",
                "dependency_suggestions",
                "license_suggestions",
            ]:
                val = getattr(se, attr, None) if se else None
                if val:
                    suggestions.append({
                        "suggestion_number": idx,
                        "category": attr,
                        "content_guidance": str(val),
                    })
                    idx += 1
            if fe is not None:
                for attr in [
                    "readability", "project_purpose", "hardware_and_software_spec",
                    "dependency", "license", "contributor_author", "overall_score",
                ]:
                    items = getattr(fe, attr, None) or []
                    for item in (items if isinstance(items, list) else [items]):
                        if item:
                            suggestions.append({
                                "suggestion_number": idx,
                                "category": attr,
                                "content_guidance": str(item),
                            })
                            idx += 1

    else:
        import json
        try:
            raw = json.dumps(eval_results, default=str)
        except Exception:
            raw = str(eval_results)
        suggestions.append({
            "suggestion_number": 1,
            "category": "general",
            "content_guidance": raw,
        })
        idx = 2

    n = len(suggestions)
    return {
        "total_suggestions": n,
        "integration_instruction": (
            f"Integrate ALL {n} suggestions below into ONE cohesive document. "
            f"Do NOT create {n} separate versions."
        ),
        "suggestions": suggestions,
    }
