"""
BioGuider pipeline tests: matrix and smoke.
"""
import os
import json
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict
import logging

import pytest
from langchain_openai import ChatOpenAI

from benchmark.shared import *

logger = logging.getLogger(__name__)

# ============================================================================
# BIOGUIDER PIPELINE BENCHMARK (evaluate + generate with real BioGuider logic)
# ============================================================================

def test_bioguider_pipeline_matrix(llm, test_output_dir):
    """
    BioGuider-pipeline benchmark on R documentation files (man/*.Rd).

    Target: 10 Seurat .Rd files selected by FileSelector (skips stub files
    that lack \\description{} or \\arguments{}).

    Injection: deterministic, category-agnostic — but because LLMErrorInjector
    detects .Rd files via _is_rd_file(), it automatically fires the two
    code-consistency categories for .Rd prose:
      - rd_func_name  transposed chars in \\code{FuncName} / \\link{FuncName}
      - rd_arg_name   transposed chars in \\code{argname} / \\item{arg} prose

    Matrix: 10 files × STRESS_LEVELS × N models.

    Pipeline:
      - DocumentPipeline.prepare_repo() called once (builds CodeStructureDb /
        SummarizedFilesDb from the Seurat repo, LLM-independent).
      - evaluate_and_refine_document(eval_type=USERGUIDE) per model per cell —
        runs EvaluationUserGuideTask then LLMContentGenerator.

    Outputs land in outputs/bioguider_rd_pipeline/run_<TS>/.

    Run:
        pytest benchmark/test_pipeline.py::test_bioguider_pipeline_matrix -v -s
    """
    import time
    from bioguider.generation.document_pipeline import DocumentPipeline
    from bioguider.utils.constants import EvaluationTypeEnum

    error_levels = STRESS_LEVELS

    # ── Select 10 largest .Rd target files ───────────────────────────────────
    rd_files = select_largest_rd_files(SEURAT_REPO_PATH, n=10)
    if not rd_files:
        pytest.skip(f"No .Rd files found under {SEURAT_REPO_PATH}/man/ — clone Seurat first")

    # ── Prepare shared pipeline once ─────────────────────────────────────────
    logger.info(f"\nPreparing shared DocumentPipeline from {SEURAT_REPO_PATH} ...")
    shared_pipeline = DocumentPipeline(SEURAT_REPO_PATH).prepare_repo(llm)
    logger.info("DocumentPipeline ready (CodeStructureDb + SummarizedFilesDb built).")

    multi_root = os.path.join(
        "outputs/bioguider_rd_pipeline",
        datetime.now().strftime("run_%Y%m%d_%H%M%S"),
    )
    os.makedirs(multi_root, exist_ok=True)

    total_cells = len(rd_files) * len(error_levels) * len(MODELS)
    logger.info(f"\n{'='*70}")
    logger.info("BIOGUIDER PIPELINE MATRIX  (man/*.Rd targets)")
    logger.info(f"{'='*70}")
    logger.info(f"Files: {len(rd_files)}, Levels: {len(error_levels)}, "
                f"Models: {len(MODELS)}, Total cells: {total_cells}")
    logger.info(f"Target files: {[Path(f).name for f in rd_files]}")
    logger.info(f"Output root:  {multi_root}")

    all_file_results: Dict[str, List[StressLevelResult]] = {}

    for test_file in rd_files:
        if not os.path.exists(test_file):
            logger.info(f"  SKIP missing file: {test_file}")
            continue

        file_stem = Path(test_file).stem
        file_out = os.path.join(multi_root, file_stem)
        os.makedirs(file_out, exist_ok=True)
        original_content = Path(test_file).read_text(encoding="utf-8")
        if not original_content.strip():
            logger.info(f"  SKIP empty file: {test_file}")
            continue

        write_file(os.path.join(file_out, f"{file_stem}.original.Rd"), original_content)

        logger.info(f"\n{'#'*70}")
        logger.info(f"# FILE: {file_stem}.Rd")
        logger.info(f"{'#'*70}")

        file_results: List[StressLevelResult] = []

        for error_level in error_levels:
            logger.info(f"\n--- Level {error_level} ---")

            # Deterministic injection — _is_rd_file() triggers rd_func_name /
            # rd_arg_name supplement loops automatically.
            injector = LLMErrorInjector(llm, force_deterministic=True)
            corrupted, manifest = injector.inject(
                original_content,
                min_per_category=error_level,
            )
            # Report how many rd_reference errors were injected
            rd_errors = [e for e in manifest.get("errors", [])
                         if e["category"] in ("rd_func_name", "rd_arg_name")]
            total_injected = len(manifest.get("errors", []))
            logger.info(f"    Injected {total_injected} errors total "
                        f"({len(rd_errors)} rd_reference: "
                        f"{sum(1 for e in rd_errors if e['category']=='rd_func_name')} rd_func_name, "
                        f"{sum(1 for e in rd_errors if e['category']=='rd_arg_name')} rd_arg_name)")

            corrupted_filename = f"{file_stem}.level_{error_level}.corrupted.Rd"
            write_file(os.path.join(file_out, corrupted_filename), corrupted)
            manifest_path = os.path.join(file_out, f"{file_stem}.level_{error_level}.manifest.json")
            with open(manifest_path, "w") as f:
                json.dump(manifest, f, indent=2)

            def _run_one_pipeline(model_name: str):
                combo = f"{model_name}+bioguider_pipeline"
                t0 = time.time()
                try:
                    model_config = MODELS.get(model_name, {"type": "litellm", "model": model_name})
                    model_id = model_config.get("model", model_name)
                    model_type = model_config.get("type", "litellm")
                    if model_type == "anthropic":
                        from langchain_anthropic import ChatAnthropic
                        model_llm = ChatAnthropic(
                            model=model_id,
                            api_key=os.environ.get("CLAUDE_API_KEY"),
                            timeout=300,
                            max_retries=1,
                            max_tokens=8192,
                        )
                    else:
                        model_llm = ChatOpenAI(
                            model=model_id,
                            api_key=os.environ.get("OPENAI_API_KEY"),
                            base_url=os.environ.get("OPENAI_BASE_URL"),
                            timeout=300,
                            max_retries=1,
                        )

                    report_path = os.path.join(
                        file_out,
                        f"{file_stem}.level_{error_level}.{model_name}.eval_report.json",
                    )
                    _, fixed_content = shared_pipeline.evaluate_and_refine_document(
                        llm=model_llm,
                        doc_repo_path=file_out,
                        doc_path=corrupted_filename,
                        eval_type=EvaluationTypeEnum.USERGUIDE,
                        report_output_path=report_path,
                    )
                    duration = time.time() - t0

                    write_file(
                        os.path.join(file_out, f"{file_stem}.level_{error_level}.{model_name}.fixed.Rd"),
                        fixed_content,
                    )

                    result, category_results = evaluate_fixes(
                        original_content,
                        corrupted,
                        fixed_content,
                        manifest,
                        llm,
                    )
                    sr = StressLevelResult(
                        error_count=error_level,
                        total_errors_injected=total_injected,
                        errors_fixed=result.true_positives,
                        errors_unfixed=result.false_negatives,
                        fix_rate=result.fix_rate,
                        precision=result.precision,
                        recall=result.recall,
                        f1_score=result.f1_score,
                        duration_seconds=duration,
                        category_results=category_results,
                        model_name=combo,
                        false_positives=getattr(result, "false_positives", 0),
                    )
                    file_results.append(sr)
                    logger.info(
                        f"    {combo:<40} F1={result.f1_score:.3f} "
                        f"fix={result.fix_rate:.1%} time={duration:.1f}s"
                    )
                except Exception as e:  # noqa: BLE001
                    logger.error(f"    {combo:<40} ERROR: {e}")

            with ThreadPoolExecutor(max_workers=len(MODELS)) as pool:
                futures = [pool.submit(_run_one_pipeline, m) for m in MODELS]
                for _ in as_completed(futures):
                    pass

        save_results(file_results, file_out)
        all_file_results[file_stem] = file_results

    # ── Cross-file aggregate ──────────────────────────────────────────────────
    pooled: List[StressLevelResult] = []
    for results in all_file_results.values():
        pooled.extend(results)

    agg_dir = os.path.join(multi_root, "_aggregate")
    os.makedirs(agg_dir, exist_ok=True)
    save_results(pooled, agg_dir)
    for old_name, new_name in [
        ("STRESS_TEST_RESULTS.json", "AGGREGATE_RESULTS.json"),
        ("STRESS_TEST_TABLE.csv", "AGGREGATE_TABLE.csv"),
        ("STRESS_TEST_CATEGORY_DETAIL.csv", "AGGREGATE_CATEGORY_DETAIL.csv"),
        ("STRESS_TEST_REPORT.md", "AGGREGATE_REPORT.md"),
    ]:
        src = os.path.join(agg_dir, old_name)
        dst = os.path.join(agg_dir, new_name)
        if os.path.exists(src):
            os.rename(src, dst)
    logger.info(f"\n{'='*70}")
    logger.info("BIOGUIDER RD PIPELINE MATRIX COMPLETE")
    logger.info(f"{'='*70}")
    logger.info(f"Files processed: {len(all_file_results)}")
    logger.info(f"Total results:   {len(pooled)}")
    logger.info(f"Artifacts:       {multi_root}")

    assert len(pooled) > 0, "No results produced — check LLM/proxy connectivity"


# ============================================================================
# BIOGUIDER PIPELINE MINIMAL (one .Rd file, quick levels, 3 models)
# ============================================================================

def test_bioguider_pipeline_matrix_minimal(llm, test_output_dir):
    """
    Minimal BioGuider-pipeline benchmark on a single .Rd file.

    Picks the largest substantive .Rd file (by byte size) to maximise injection
    yield for rd_func_name / rd_arg_name.  Runs at QUICK_STRESS_LEVELS
    (10 / 40 / 100 errors/category) across three models in parallel.

    Injection: deterministic. _is_rd_file() triggers rd_func_name / rd_arg_name
    supplement loops automatically alongside any other anchored categories.
    Pipeline: evaluate_and_refine_document(eval_type=USERGUIDE) per model.

    Outputs land in outputs/bioguider_rd_pipeline_minimal/run_<TS>/.

    Run:
        pytest benchmark/test_pipeline.py::test_bioguider_pipeline_matrix_minimal -v -s
    """
    import time
    from bioguider.generation.document_pipeline import DocumentPipeline
    from bioguider.utils.constants import EvaluationTypeEnum

    TARGET_MODELS = ["gpt-4o", "gpt-5.4", "kimi-k2.5"]
    error_levels = QUICK_STRESS_LEVELS

    # ── Pick the single largest .Rd file ─────────────────────────────────────
    rd_files = select_largest_rd_files(SEURAT_REPO_PATH, n=1)
    if not rd_files:
        pytest.skip(f"No .Rd files found under {SEURAT_REPO_PATH}/man/ — clone Seurat first")
    test_file = rd_files[0]
    file_stem = Path(test_file).stem

    # ── Prepare shared pipeline once ─────────────────────────────────────────
    logger.info(f"\nPreparing shared DocumentPipeline from {SEURAT_REPO_PATH} ...")
    shared_pipeline = DocumentPipeline(SEURAT_REPO_PATH).prepare_repo(llm)
    logger.info("DocumentPipeline ready.")

    run_root = os.path.join(
        "outputs/bioguider_rd_pipeline_minimal",
        datetime.now().strftime("run_%Y%m%d_%H%M%S"),
    )
    os.makedirs(run_root, exist_ok=True)

    original_content = Path(test_file).read_text(encoding="utf-8")
    assert original_content.strip(), f"Empty .Rd file: {test_file}"
    write_file(os.path.join(run_root, f"{file_stem}.original.Rd"), original_content)

    logger.info(f"\n{'='*70}")
    logger.info(f"BIOGUIDER RD PIPELINE MINIMAL  file={file_stem}.Rd")
    logger.info(f"{'='*70}")
    logger.info(f"Models: {TARGET_MODELS}  Levels: {error_levels}")
    logger.info(f"Output: {run_root}")

    file_results: List[StressLevelResult] = []

    for error_level in error_levels:
        logger.info(f"\n--- Level {error_level} ---")

        injector = LLMErrorInjector(llm, force_deterministic=True)
        corrupted, manifest = injector.inject(original_content, min_per_category=error_level)
        total_injected = len(manifest.get("errors", []))
        rd_errors = [e for e in manifest.get("errors", [])
                     if e["category"] in ("rd_func_name", "rd_arg_name")]
        logger.info(f"    Injected {total_injected} errors "
                    f"({len(rd_errors)} rd_reference: "
                    f"{sum(1 for e in rd_errors if e['category']=='rd_func_name')} rd_func_name, "
                    f"{sum(1 for e in rd_errors if e['category']=='rd_arg_name')} rd_arg_name)")

        corrupted_filename = f"{file_stem}.level_{error_level}.corrupted.Rd"
        write_file(os.path.join(run_root, corrupted_filename), corrupted)
        with open(os.path.join(run_root, f"{file_stem}.level_{error_level}.manifest.json"), "w") as f:
            json.dump(manifest, f, indent=2)

        def _run_one_model(model_name: str):
            combo = f"{model_name}+bioguider"
            t0 = time.time()
            try:
                model_config = MODELS.get(model_name, {"type": "litellm", "model": model_name})
                model_id = model_config.get("model", model_name)
                model_type = model_config.get("type", "litellm")
                if model_type == "anthropic":
                    from langchain_anthropic import ChatAnthropic
                    model_llm = ChatAnthropic(
                        model=model_id,
                        api_key=os.environ.get("CLAUDE_API_KEY"),
                        timeout=300,
                        max_retries=1,
                        max_tokens=8192,
                    )
                else:
                    model_llm = ChatOpenAI(
                        model=model_id,
                        api_key=os.environ.get("OPENAI_API_KEY"),
                        base_url=os.environ.get("OPENAI_BASE_URL"),
                        timeout=300,
                        max_retries=1,
                    )

                report_path = os.path.join(
                    run_root,
                    f"{file_stem}.level_{error_level}.{model_name}.eval_report.json",
                )
                _, fixed_content = shared_pipeline.evaluate_and_refine_document(
                    llm=model_llm,
                    doc_repo_path=run_root,
                    doc_path=corrupted_filename,
                    eval_type=EvaluationTypeEnum.USERGUIDE,
                    report_output_path=report_path,
                )
                duration = time.time() - t0

                write_file(
                    os.path.join(run_root, f"{file_stem}.level_{error_level}.{model_name}.fixed.Rd"),
                    fixed_content,
                )

                result, category_results = evaluate_fixes(
                    original_content, corrupted, fixed_content, manifest, llm,
                )
                sr = StressLevelResult(
                    error_count=error_level,
                    total_errors_injected=total_injected,
                    errors_fixed=result.true_positives,
                    errors_unfixed=result.false_negatives,
                    fix_rate=result.fix_rate,
                    precision=result.precision,
                    recall=result.recall,
                    f1_score=result.f1_score,
                    duration_seconds=duration,
                    category_results=category_results,
                    model_name=combo,
                    false_positives=getattr(result, "false_positives", 0),
                )
                logger.info(
                    f"    {combo:<36} F1={result.f1_score:.3f} "
                    f"fix={result.fix_rate:.1%} time={duration:.1f}s"
                )
                return sr
            except Exception as e:  # noqa: BLE001
                logger.error(f"    {combo:<36} ERROR: {e}")
                return None

        with ThreadPoolExecutor(max_workers=len(TARGET_MODELS)) as pool:
            futures = {pool.submit(_run_one_model, m): m for m in TARGET_MODELS}
            for future in as_completed(futures):
                sr = future.result()
                if sr is not None:
                    file_results.append(sr)

    save_results(file_results, run_root)
    for old_name, new_name in [
        ("STRESS_TEST_RESULTS.json", "RESULTS.json"),
        ("STRESS_TEST_TABLE.csv", "TABLE.csv"),
        ("STRESS_TEST_CATEGORY_DETAIL.csv", "CATEGORY_DETAIL.csv"),
        ("STRESS_TEST_REPORT.md", "REPORT.md"),
    ]:
        src = os.path.join(run_root, old_name)
        dst = os.path.join(run_root, new_name)
        if os.path.exists(src):
            os.rename(src, dst)

    logger.info(f"\n{'='*70}")
    logger.info("RESULTS SUMMARY")
    logger.info(f"{'='*70}")
    logger.info(f"{'Model':<36} | {'Level':>5} | {'F1':>6} | {'Fix%':>6} | {'Time':>7}")
    logger.info("-" * 70)
    for sr in sorted(file_results, key=lambda r: (r.model_name, r.error_count)):
        logger.info(
            f"{sr.model_name:<36} | {sr.error_count:>5} | {sr.f1_score:>6.3f} "
            f"| {sr.fix_rate:>5.1%} | {sr.duration_seconds:>6.1f}s"
        )
    logger.info(f"\nArtifacts: {run_root}")

    assert file_results, "No results produced — check LLM/proxy connectivity"


# ============================================================================
# EVALUATE AND REFINE A PRE-EXISTING CORRUPTED FILE
# ============================================================================

def test_evaluation_and_refine_corrupt_file(llm, test_output_dir):
    """
    Evaluate and refine a pre-existing corrupted .Rd file produced by a
    previous benchmark run.

    The default target is the level-10 corrupted FindMarkers.Rd from the most
    recent minimal-pipeline run.  Override via the CORRUPT_FILE env var:

        CORRUPT_FILE=outputs/bioguider_rd_pipeline_minimal/run_XYZ/FindMarkers.level_40.corrupted.Rd \\
            pytest benchmark/test_pipeline.py::test_evaluation_and_refine_corrupt_file -v -s

    Steps:
      1. Locate the corrupted file and its companion manifest (.manifest.json).
      2. Build the shared DocumentPipeline (prepare_repo) from SEURAT_REPO_PATH.
      3. Run evaluate_and_refine_document() with eval_type=USERGUIDE.
      4. Write the refined file and evaluation report to test_output_dir.
      5. If the manifest exists, score the fix and print per-category results.

    Run:
        pytest benchmark/test_pipeline.py::test_evaluation_and_refine_corrupt_file -v -s
    """
    import time
    from bioguider.generation.document_pipeline import DocumentPipeline
    from bioguider.utils.constants import EvaluationTypeEnum

    # ── Locate the corrupted file ─────────────────────────────────────────────
    default_corrupt = (
        "outputs/bioguider_rd_pipeline_minimal/run_20260505_150324/"
        "FindMarkers.level_10.corrupted.Rd"
    )
    corrupt_path = os.environ.get("CORRUPT_FILE", default_corrupt)

    if not os.path.exists(corrupt_path):
        pytest.skip(
            f"Corrupted file not found: {corrupt_path}\n"
            "Run test_bioguider_pipeline_matrix_minimal first, or set "
            "CORRUPT_FILE=<path> to a different corrupted file."
        )

    corrupt_path = os.path.abspath(corrupt_path)
    doc_repo_path = os.path.dirname(corrupt_path)
    doc_filename = os.path.basename(corrupt_path)
    file_stem = Path(corrupt_path).stem  # e.g. FindMarkers.level_10.corrupted

    # Companion manifest: same name with .manifest.json suffix
    # e.g. FindMarkers.level_10.corrupted.Rd → FindMarkers.level_10.manifest.json
    # Convention used by test_bioguider_pipeline_matrix_minimal:
    #   FindMarkers.level_10.manifest.json
    base_parts = doc_filename.rsplit(".", 3)  # ['FindMarkers', 'level_10', 'corrupted', 'Rd']
    if len(base_parts) >= 3 and base_parts[-2] == "corrupted":
        manifest_name = ".".join(base_parts[:-2]) + ".manifest.json"
    else:
        manifest_name = doc_filename.replace(".corrupted.Rd", ".manifest.json")
    manifest_path = os.path.join(doc_repo_path, manifest_name)

    original_content = Path(corrupt_path).read_text(encoding="utf-8", errors="replace")
    assert original_content.strip(), f"Corrupted file is empty: {corrupt_path}"

    logger.info(f"\nCorrupted file : {corrupt_path}")
    logger.info(f"Manifest       : {manifest_path if os.path.exists(manifest_path) else 'NOT FOUND (scoring skipped)'}")

    # ── Build DocumentPipeline ────────────────────────────────────────────────
    logger.info(f"\nPreparing DocumentPipeline from {SEURAT_REPO_PATH} ...")
    pipeline = DocumentPipeline(SEURAT_REPO_PATH).prepare_repo(llm)
    logger.info("DocumentPipeline ready.")

    # ── Evaluate + refine ─────────────────────────────────────────────────────
    report_out = os.path.join(test_output_dir, f"{file_stem}.eval_report.json")
    t0 = time.time()
    model_name = "gpt-oss"
    model_config = MODELS.get(model_name, {"type": "litellm", "model": model_name})
    model_id = model_config.get("model", model_name)
    model_type = model_config.get("type", "litellm")
    if model_type == "anthropic":
        from langchain_anthropic import ChatAnthropic
        model_llm = ChatAnthropic(
            model=model_id,
            api_key=os.environ.get("CLAUDE_API_KEY"),
            timeout=300,
            max_retries=1,
            max_tokens=8192,
        )
    else:
        model_llm = ChatOpenAI(
            model=model_id,
            api_key=os.environ.get("OPENAI_API_KEY"),
            base_url=os.environ.get("OPENAI_BASE_URL"),
            timeout=300,
            max_retries=1,
        )
    merged_report, refined_content = pipeline.evaluate_and_refine_document(
        llm=model_llm,
        doc_repo_path=doc_repo_path,
        doc_path=doc_filename,
        eval_type=EvaluationTypeEnum.USERGUIDE,
        report_output_path=report_out,
    )
    duration = time.time() - t0

    n_suggestions = merged_report.get("total_suggestions", 0)
    logger.info(f"\nEvaluation + refinement done in {duration:.1f}s")
    logger.info(f"Suggestions found: {n_suggestions}")
    logger.info(f"Eval report      : {report_out}")

    # Write refined file
    refined_path = os.path.join(test_output_dir, f"{file_stem}.refined.Rd")
    write_file(refined_path, refined_content)
    logger.info(f"Refined file     : {refined_path}")

    # ── Score against manifest if available ───────────────────────────────────
    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            manifest = json.load(f)

        # We need the "original" (pre-corruption) content to score.
        # Look for the companion .original.Rd file in the same directory.
        original_stem = doc_filename.split(".level_")[0]  # 'FindMarkers'
        original_rd = os.path.join(doc_repo_path, f"{original_stem}.original.Rd")
        if os.path.exists(original_rd):
            pre_corrupt = Path(original_rd).read_text(encoding="utf-8", errors="replace")
        else:
            # Fall back to SEURAT_REPO_PATH/man/<stem>.Rd
            fallback = os.path.join(SEURAT_REPO_PATH, "man", f"{original_stem}.Rd")
            if os.path.exists(fallback):
                pre_corrupt = Path(fallback).read_text(encoding="utf-8", errors="replace")
                logger.info(f"Using fallback original: {fallback}")
            else:
                pre_corrupt = None
                logger.warning("Original (pre-corruption) file not found — skipping scoring")

        if pre_corrupt is not None:
            result, category_results = evaluate_fixes(
                pre_corrupt, original_content, refined_content, manifest, llm,
            )
            logger.info(f"\n{'='*60}")
            logger.info("SCORING RESULTS")
            logger.info(f"{'='*60}")
            logger.info(
                f"F1={result.f1_score:.3f}  Fix%={result.fix_rate:.1%}  "
                f"Precision={result.precision:.3f}  Recall={result.recall:.3f}  "
                f"FP={getattr(result, 'false_positives', 0)}"
            )
            logger.info(f"\n{'Category':<28} {'Fixed/Injected':>15}  {'Fix%':>6}")
            logger.info("-" * 55)
            for cr in sorted(category_results, key=lambda c: c.category):
                if cr.injected > 0:
                    logger.info(
                        f"{cr.category:<28} {cr.fixed:>6}/{cr.injected:<7}  {cr.fix_rate:>5.0%}"
                    )
    else:
        logger.info("\n(No manifest — scoring skipped)")

    assert refined_content.strip(), "Refined content is empty"
    assert len(refined_content) > 100, "Refined content suspiciously short"


# ============================================================================
# BIOGUIDER PIPELINE SMOKE TEST (evaluate + generate, one vignette, 3 models)
# ============================================================================

def test_bioguider_pipeline_smoke(llm, test_output_dir):
    """
    Smoke test for the full BioGuider evaluate-then-generate pipeline.

    Runs one vignette (cell_cycle_vignette.Rmd) at a single error level (10)
    through DocumentPipeline.evaluate_and_refine_document() for each of three
    models (gpt-4o, gpt-5.4, kimi-k2.5) in parallel.

    Each cell:
      1. Injects 10 errors/category deterministically.
      2. Calls evaluate_and_refine_document() — runs EvaluationTutorialTask
         then LLMContentGenerator (2 LLM calls per model).
      3. Writes the evaluation report to disk.
      4. Scores the fixed output against the injection manifest.

    Run:
        pytest system_tests/test_single_file_stress.py::test_bioguider_pipeline_smoke -v -s
    """
    import time
    from bioguider.generation.document_pipeline import DocumentPipeline
    from bioguider.utils.constants import EvaluationTypeEnum

    TARGET_MODELS = ["gpt-4o", "gpt-5.4", "kimi-k2.5"]
    ERROR_LEVEL = 10
    TEST_FILE = f"{SEURAT_VIGNETTES_DIR}/cell_cycle_vignette.Rmd"

    if not os.path.exists(TEST_FILE):
        import pytest
        pytest.skip(f"Vignette not found: {TEST_FILE}")

    run_root = os.path.join(
        "outputs/bioguider_pipeline_smoke",
        datetime.now().strftime("run_%Y%m%d_%H%M%S"),
    )
    os.makedirs(run_root, exist_ok=True)

    file_stem = Path(TEST_FILE).stem
    original_content = read_file(TEST_FILE) or ""
    assert original_content.strip(), f"Empty vignette: {TEST_FILE}"

    # Save original for reference
    write_file(os.path.join(run_root, f"{file_stem}.original.Rmd"), original_content)

    # ── Shared pipeline (built once, reused across models) ─────────────────
    logger.info(f"\nBuilding DocumentPipeline from {SEURAT_REPO_PATH} ...")
    shared_pipeline = DocumentPipeline(SEURAT_REPO_PATH).prepare_repo(llm)
    logger.info("DocumentPipeline ready.")

    # ── Deterministic injection (same corrupted doc for every model) ───────
    injector = LLMErrorInjector(llm, force_deterministic=True)
    corrupted, manifest = injector.inject(
        original_content,
        min_per_category=ERROR_LEVEL,
        max_words=50000,
    )
    total_injected = len(manifest.get("errors", []))
    logger.info(f"Injected {total_injected} errors at level={ERROR_LEVEL}")

    corrupted_path = os.path.join(run_root, f"{file_stem}.level_{ERROR_LEVEL}.corrupted.Rmd")
    write_file(corrupted_path, corrupted)
    corrupted_filename = f"{file_stem}.level_{ERROR_LEVEL}.corrupted.Rmd"
    with open(os.path.join(run_root, f"{file_stem}.level_{ERROR_LEVEL}.manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    # ── Per-model fix + score ──────────────────────────────────────────────
    def _run_one_model(model_name: str):
        combo = f"{model_name}+bioguider"
        t0 = time.time()
        try:
            model_config = MODELS.get(model_name, {"type": "litellm", "model": model_name})
            model_id = model_config.get("model", model_name)
            model_type = model_config.get("type", "litellm")
            if model_type == "anthropic":
                from langchain_anthropic import ChatAnthropic
                model_llm = ChatAnthropic(
                    model=model_id,
                    api_key=os.environ.get("CLAUDE_API_KEY"),
                    timeout=300,
                    max_retries=1,
                    max_tokens=8192,
                )
            else:
                model_llm = ChatOpenAI(
                    model=model_id,
                    api_key=os.environ.get("OPENAI_API_KEY"),
                    base_url=os.environ.get("OPENAI_BASE_URL"),
                    timeout=300,
                    max_retries=1,
                )

            report_path = os.path.join(
                run_root,
                f"{file_stem}.level_{ERROR_LEVEL}.{model_name}.eval_report.json",
            )

            merged_report, fixed_content = shared_pipeline.evaluate_and_refine_document(
                llm=model_llm,
                doc_repo_path=run_root,
                doc_path=corrupted_filename,
                eval_type=EvaluationTypeEnum.TUTORIAL,
                report_output_path=report_path,
            )
            duration = time.time() - t0

            suggestions_count = merged_report.get("total_suggestions", 0)
            logger.info(f"  [{combo}] eval_report: {suggestions_count} suggestions → {report_path}")

            fixed_path = os.path.join(
                run_root,
                f"{file_stem}.level_{ERROR_LEVEL}.{model_name}.fixed.Rmd",
            )
            write_file(fixed_path, fixed_content)

            result, category_results = evaluate_fixes(
                original_content,
                corrupted,
                fixed_content,
                manifest,
                llm,
            )

            logger.info(
                f"  [{combo}] F1={result.f1_score:.3f} "
                f"FixRate={result.fix_rate:.3f} "
                f"duration={duration:.1f}s"
            )
            for cr in category_results:
                if cr.injected > 0:
                    logger.info(f"    {cr.category}: {cr.fixed}/{cr.injected} fixed ({cr.fix_rate:.0%})")

            return combo, result, category_results, duration, suggestions_count

        except Exception as exc:
            duration = time.time() - t0
            logger.error(f"  [{combo}] FAILED after {duration:.1f}s: {exc}")
            raise

    logger.info(f"\n{'='*70}")
    logger.info(f"BIOGUIDER PIPELINE SMOKE  file={file_stem}  level={ERROR_LEVEL}")
    logger.info(f"{'='*70}")
    logger.info(f"Models: {TARGET_MODELS}")

    results = {}
    with ThreadPoolExecutor(max_workers=len(TARGET_MODELS)) as executor:
        futures = {executor.submit(_run_one_model, m): m for m in TARGET_MODELS}
        for future in as_completed(futures):
            model = futures[future]
            try:
                combo, result, cat_results, dur, n_suggestions = future.result()
                results[combo] = {
                    "f1": result.f1_score,
                    "fix_rate": result.fix_rate,
                    "precision": result.precision,
                    "recall": result.recall,
                    "false_positives": getattr(result, "false_positives", 0),
                    "duration_s": dur,
                    "suggestions": n_suggestions,
                }
            except Exception as exc:
                results[f"{model}+bioguider"] = {"error": str(exc)}

    # ── Summary ────────────────────────────────────────────────────────────
    logger.info(f"\n{'='*70}")
    logger.info("RESULTS SUMMARY")
    logger.info(f"{'='*70}")
    logger.info(f"{'Model':<32} | {'F1':>6} | {'Fix%':>6} | {'Suggestions':>11} | {'Time':>7}")
    logger.info("-" * 70)
    for combo, r in sorted(results.items()):
        if "error" in r:
            logger.info(f"{combo:<32} | ERROR: {r['error']}")
        else:
            logger.info(
                f"{combo:<32} | {r['f1']:>6.3f} | {r['fix_rate']:>5.1%} "
                f"| {r['suggestions']:>11} | {r['duration_s']:>6.1f}s"
            )

    summary_path = os.path.join(run_root, "SMOKE_TEST_RESULTS.json")
    with open(summary_path, "w") as f:
        json.dump({
            "file": file_stem,
            "error_level": ERROR_LEVEL,
            "total_injected": total_injected,
            "models": TARGET_MODELS,
            "results": results,
        }, f, indent=2)
    logger.info(f"\nArtifacts: {run_root}")

    successful = [r for r in results.values() if "error" not in r]
    assert successful, "All models failed — check LLM/proxy connectivity"
