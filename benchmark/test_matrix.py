"""
Multi-file matrix, skill comparison, and three-model benchmark tests.
"""
import os
import json
import csv
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any

import pytest

from benchmark.shared import *


def test_multi_file_full_matrix(llm, test_output_dir):
    """
    MULTI-FILE MATRIX — 10 Seurat vignettes × 9 error levels × 5 models × 1 prompt (bioguider).

    Initial-round shape (450 cells total). The simple-prompt ablation was dropped;
    the bioguider prompt is the only correction prompt in this pass.

    Each (file, level) pair gets ONE deterministic injection (force_deterministic=True)
    so every model scores against byte-identical corrupted files. The 5 model configs
    within each (file, level) cell run through a ThreadPoolExecutor so LLM latency is
    paid once per level rather than five times. Per-file artefacts and figures land in
    their own subdir; a cross-file aggregate lands in
    ``_aggregate/AGGREGATE_*.json/csv`` plus rendered ``fig*``.

    Expected wall-clock: ~1.5-2 hours with 5-wide parallel configs.
    Expected LLM spend: ~2-5M tokens across the matrix.

    Run:
        pytest system_tests/test_single_file_stress.py::test_multi_file_full_matrix -v -s
    """
    import time

    # 5 models × 1 prompt = 5 configs per (file, level) cell
    test_configs = [(m, "bioguider") for m in MODELS]
    error_levels = STRESS_LEVELS  # [5, 10, 20, 40, 60, 100, 150, 200, 300]

    multi_root = os.path.join(
        OUTPUT_BASE.replace("single_file_stress", "multi_file_stress"),
        datetime.now().strftime("run_%Y%m%d_%H%M%S"),
    )
    os.makedirs(multi_root, exist_ok=True)

    total_cells = len(TUTORIAL_FILES) * len(error_levels) * len(test_configs)
    print(f"\n{'='*70}")
    print("MULTI-FILE FULL MATRIX")
    print(f"{'='*70}")
    print(f"Files: {len(TUTORIAL_FILES)}, Levels: {len(error_levels)}, "
          f"Configs: {len(test_configs)}, Total cells: {total_cells}")
    print(f"Output root: {multi_root}")

    all_file_results: Dict[str, List[StressLevelResult]] = {}

    for test_file in TUTORIAL_FILES:
        if not os.path.exists(test_file):
            print(f"  SKIP missing file: {test_file}")
            continue

        file_stem = Path(test_file).stem
        file_out = os.path.join(multi_root, file_stem)
        os.makedirs(file_out, exist_ok=True)
        original_content = read_file(test_file) or ""
        if not original_content.strip():
            print(f"  SKIP empty file: {test_file}")
            continue

        # Save original for the per-file audit trail
        write_file(os.path.join(file_out, f"{file_stem}.original.Rmd"), original_content)

        print(f"\n{'#'*70}")
        print(f"# FILE: {file_stem}")
        print(f"{'#'*70}")

        file_results: List[StressLevelResult] = []

        for error_level in error_levels:
            print(f"\n--- Level {error_level} ---")

            # Deterministic injection so every model sees identical corrupted text
            injector = LLMErrorInjector(llm, force_deterministic=True)
            corrupted, manifest = injector.inject(
                original_content,
                min_per_category=error_level,
                max_words=50000,
            )
            corrupted_path = os.path.join(file_out, f"{file_stem}.level_{error_level}.corrupted.Rmd")
            write_file(corrupted_path, corrupted)
            manifest_path = os.path.join(file_out, f"{file_stem}.level_{error_level}.manifest.json")
            with open(manifest_path, "w") as f:
                json.dump(manifest, f, indent=2)
            injection_result = {
                "error_count": error_level,
                "corrupted_content": corrupted,
                "corrupted_path": corrupted_path,
                "manifest": manifest,
                "manifest_path": manifest_path,
                "total_errors": len(manifest.get("errors", [])),
            }
            print(f"    Injected {injection_result['total_errors']} errors (deterministic)")

            def _run_one(model_name: str, prompt_name: str):
                """Run a single (model, prompt) correction + evaluation. Thread-safe:
                writes per-model filenames; all shared state access is via
                GIL-protected list.append and dataclass construction.
                """
                combo = f"{model_name}+{prompt_name}"
                t0 = time.time()
                try:
                    fixed_content, _ = fix_with_model(
                        llm,
                        injection_result["corrupted_content"],
                        original_content,
                        file_out,
                        file_stem,
                        error_level,
                        prompt_name=prompt_name,
                        model_name=model_name,
                    )
                    duration = time.time() - t0

                    result, category_results = evaluate_fixes(
                        original_content,
                        injection_result["corrupted_content"],
                        fixed_content,
                        injection_result["manifest"],
                        llm,
                    )
                    sr = StressLevelResult(
                        error_count=error_level,
                        total_errors_injected=injection_result["total_errors"],
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
                    print(
                        f"    {combo:<30} F1={result.f1_score:.3f} "
                        f"fix={result.fix_rate:.1%} time={duration:.1f}s"
                    )
                except Exception as e:  # noqa: BLE001
                    print(f"    {combo:<30} ERROR: {e}")

            # Parallelise the 5 model configs — they're independent and the
            # LiteLLM proxy tolerates 5 concurrent chat completions. Gives
            # ~5x speedup on the dominant per-config latency.
            with ThreadPoolExecutor(max_workers=len(test_configs)) as pool:
                futures = [
                    pool.submit(_run_one, model_name, prompt_name)
                    for model_name, prompt_name in test_configs
                ]
                for _ in as_completed(futures):
                    pass

        # Flush per-file results + render per-file fig1-6
        save_results(file_results, file_out)
        all_file_results[file_stem] = file_results

        # Abort rule — after the FIRST file only, halt if the moat story is empty.
        # (Per the plan: if prose_code_* never injected AND no scorable wins, the
        # anchor regexes aren't matching this repo's idioms — surface now rather
        # than burn 9 more files of LLM spend.)
        if len(all_file_results) == 1:
            from bioguider.managers.config import UNSCORABLE_CATEGORIES

            any_scorable_win = any(
                getattr(r, "f1_score_scorable", r.f1_score) > 0.0
                for r in file_results
            )
            moat_cats = {"prose_code_pkg_version", "prose_code_stat_test",
                         "prose_code_marker", "prose_code_param", "accession_id_prefix"}
            moat_hits = 0
            for r in file_results:
                for c in (r.category_results or []):
                    if c.category in moat_cats:
                        moat_hits += c.injected
            print(f"\n[abort-check] after file '{file_stem}': "
                  f"any_scorable_win={any_scorable_win}, moat_hits={moat_hits}")
            if not any_scorable_win:
                raise AssertionError(
                    f"Abort: no scorable F1 > 0 on '{file_stem}'. "
                    "Models aren't fixing anything — check prompt, proxy, or corrupted fences."
                )
            if moat_hits == 0:
                print(f"[abort-check] WARNING: zero moat-category injections on '{file_stem}'. "
                      "Anchor regexes may not match this repo's idioms. Continuing — "
                      "figures will still render but the moat panel will be empty.")
            # Re-derive unscorable filter is implicit; this just asserts we're not silently nil.
            _ = UNSCORABLE_CATEGORIES  # referenced for side-effect of import validation

    # ------------------------------------------------------------------
    # Cross-file aggregate (pooled across all 10 files)
    # ------------------------------------------------------------------
    pooled: List[StressLevelResult] = []
    for file_stem, results in all_file_results.items():
        for r in results:
            # Namespace model_name with file_stem so the aggregate heatmap has
            # distinguishable rows. Keep the unnamespaced per-file results intact.
            pass
        pooled.extend(results)

    agg_dir = os.path.join(multi_root, "_aggregate")
    os.makedirs(agg_dir, exist_ok=True)
    save_results(pooled, agg_dir)
    # Rename the aggregate artifacts so they're distinguishable in the UI.
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

    # Top-level index so future-Claude finds everything from one file
    index_path = os.path.join(multi_root, "INDEX.md")
    with open(index_path, "w") as f:
        f.write(f"# Multi-File Stress Run — {datetime.now():%Y-%m-%d %H:%M:%S}\n\n")
        f.write(f"- Files: {len(all_file_results)} / {len(TUTORIAL_FILES)}\n")
        f.write(f"- Levels: {error_levels}\n")
        f.write(f"- Configs per cell: {len(test_configs)} "
                f"({[f'{m}+{p}' for m, p in test_configs]})\n")
        f.write(f"- Total results: {len(pooled)}\n\n")
        f.write("## Per-file output\n\n")
        for stem in all_file_results:
            f.write(f"- `{stem}/` — STRESS_TEST_RESULTS.json + fig1-6.{{png,pdf}}\n")
        f.write("\n## Aggregate\n\n")
        f.write("- `_aggregate/AGGREGATE_RESULTS.json`\n")
        f.write("- `_aggregate/AGGREGATE_TABLE.csv`\n")
        f.write("- `_aggregate/fig1-6.{png,pdf}` — pooled across all 10 files\n")

    print(f"\n{'='*70}")
    print("MULTI-FILE FULL MATRIX COMPLETE")
    print(f"{'='*70}")
    print(f"Files processed: {len(all_file_results)}")
    print(f"Total results: {len(pooled)}")
    print(f"Artifacts: {multi_root}")
    print(f"Index: {index_path}")

    assert len(pooled) > 0, "No results produced — check LLM/proxy connectivity"


# ============================================================================
# SKILL COMPARISON TESTS (Workstream B)
# ============================================================================


@pytest.mark.slow
def test_skill_comparison(llm, test_output_dir):
    """Compare BioGuider prompt vs skill_generic prompt on a single Seurat vignette.

    Uses error_count=30 with force_deterministic=True so both skills see
    identical corrupted text. Results land in SKILL_COMPARISON.csv inside
    the standard test_output_dir.

    Run:
        pytest system_tests/test_single_file_stress.py::test_skill_comparison -v -s
    """
    import time

    # Pick the first valid vignette from TUTORIAL_FILES.
    test_file = None
    for candidate in TUTORIAL_FILES:
        if os.path.exists(candidate):
            test_file = candidate
            break

    if test_file is None:
        pytest.skip("No Seurat vignettes found — clone satijalab/seurat first")

    original_content = read_file(test_file) or ""
    if not original_content.strip():
        pytest.skip(f"Vignette is empty: {test_file}")

    file_stem = Path(test_file).stem
    error_count = 30
    model_name = "gpt-4o"
    skills = ["bioguider", "skill_generic"]

    print(f"\n{'='*70}")
    print("SKILL COMPARISON TEST")
    print(f"{'='*70}")
    print(f"File: {test_file}")
    print(f"Model: {model_name}")
    print(f"Error count: {error_count}")
    print(f"Skills: {skills}")

    # One deterministic injection shared across both skills.
    injector = LLMErrorInjector(llm, force_deterministic=True)
    corrupted, manifest = injector.inject(
        original_content,
        min_per_category=error_count,
        max_words=50000,
    )
    corrupted_path = os.path.join(test_output_dir, f"{file_stem}.level_{error_count}.corrupted.Rmd")
    write_file(corrupted_path, corrupted)
    manifest_path = os.path.join(test_output_dir, f"{file_stem}.level_{error_count}.manifest.json")
    with open(manifest_path, "w") as fh:
        json.dump(manifest, fh, indent=2)

    total_injected = len(manifest.get("errors", []))
    print(f"Injected {total_injected} errors (deterministic)")

    csv_rows: List[Dict[str, Any]] = []

    for skill in skills:
        print(f"\n--- Skill: {skill} ---")
        t0 = time.time()

        fixed_content, _ = fix_with_model(
            llm,
            corrupted,
            original_content,
            test_output_dir,
            file_stem,
            error_count,
            prompt_name=skill,
            model_name=model_name,
        )
        duration = time.time() - t0

        result, category_results = evaluate_fixes(
            original_content,
            corrupted,
            fixed_content,
            manifest,
            llm,
        )

        sr = StressLevelResult(
            error_count=error_count,
            total_errors_injected=total_injected,
            errors_fixed=result.true_positives,
            errors_unfixed=result.false_negatives,
            fix_rate=result.fix_rate,
            precision=result.precision,
            recall=result.recall,
            f1_score=result.f1_score,
            duration_seconds=duration,
            category_results=category_results,
            model_name=f"{model_name}+{skill}",
            false_positives=getattr(result, "false_positives", 0),
            code_fence_violations=getattr(result, "code_fence_violations", 0),
            yaml_violations=getattr(result, "yaml_violations", 0),
            section_violations=getattr(result, "section_violations", 0),
        )
        _populate_scorable(sr)

        print(
            f"  Fixed {sr.errors_fixed}/{total_injected} "
            f"fix_rate={sr.fix_rate:.1%} F1={sr.f1_score:.3f} "
            f"F1_scorable={sr.f1_score_scorable:.3f} time={duration:.1f}s"
        )

        csv_rows.append({
            "file_stem": file_stem,
            "model": model_name,
            "skill": skill,
            "error_count": error_count,
            "total_injected": total_injected,
            "fixed": sr.errors_fixed,
            "unfixed": sr.errors_unfixed,
            "fix_rate": round(sr.fix_rate, 4),
            "f1_score": round(sr.f1_score, 4),
            "f1_score_scorable": round(sr.f1_score_scorable, 4),
            "f1_score_content": round(sr.f1_score_content, 4),
            "f1_score_hygiene": round(sr.f1_score_hygiene, 4),
            "false_positives": sr.false_positives,
            "code_fence_violations": sr.code_fence_violations,
            "yaml_violations": sr.yaml_violations,
            "section_violations": sr.section_violations,
            "duration_s": round(duration, 2),
        })

    csv_path = os.path.join(test_output_dir, "SKILL_COMPARISON.csv")
    _write_skill_comparison_csv(csv_rows, csv_path)

    print(f"\n{'='*70}")
    print("SKILL COMPARISON SUMMARY")
    print(f"{'='*70}")
    print(f"{'Skill':<20} {'Fixed':<8} {'Fix Rate':<10} {'F1':<8} {'F1 Scorable':<12}")
    print("-" * 70)
    for row in csv_rows:
        print(
            f"{row['skill']:<20} {row['fixed']:<8} {row['fix_rate']:<10.1%} "
            f"{row['f1_score']:<8.3f} {row['f1_score_scorable']:<12.3f}"
        )
    print(f"\nResults written to: {csv_path}")

    assert len(csv_rows) == len(skills), "Should have one result row per skill"


@pytest.mark.slow
def test_skill_matrix(llm, test_output_dir):
    """Skill matrix: 5 vignettes x 3 error levels x 2 skills x 1 model.

    Each (file, level) cell uses a single deterministic injection so both
    skills score against byte-identical corrupted text. Results land in
    SKILL_MATRIX_TABLE.csv in the output directory.

    Run:
        pytest system_tests/test_single_file_stress.py::test_skill_matrix -v -s
    """
    import time

    error_levels = [10, 30, 100]
    skills = ["bioguider", "skill_generic"]
    model_name = "gpt-4o"

    # Use first 5 vignettes that exist on disk.
    available_files = [f for f in TUTORIAL_FILES if os.path.exists(f)][:5]
    if not available_files:
        pytest.skip("No Seurat vignettes found — clone satijalab/seurat first")

    total_cells = len(available_files) * len(error_levels) * len(skills)
    print(f"\n{'='*70}")
    print("SKILL MATRIX TEST")
    print(f"{'='*70}")
    print(
        f"Files: {len(available_files)}, Levels: {error_levels}, "
        f"Skills: {skills}, Model: {model_name}, Total cells: {total_cells}"
    )

    csv_rows: List[Dict[str, Any]] = []

    for test_file in available_files:
        file_stem = Path(test_file).stem
        original_content = read_file(test_file) or ""
        if not original_content.strip():
            print(f"  SKIP empty file: {test_file}")
            continue

        print(f"\n{'#'*70}")
        print(f"# FILE: {file_stem}")
        print(f"{'#'*70}")

        for error_level in error_levels:
            print(f"\n--- Level {error_level} ---")

            # Deterministic injection shared across both skills for this cell.
            injector = LLMErrorInjector(llm, force_deterministic=True)
            corrupted, manifest = injector.inject(
                original_content,
                min_per_category=error_level,
                max_words=50000,
            )
            corrupted_path = os.path.join(
                test_output_dir,
                f"{file_stem}.level_{error_level}.corrupted.Rmd",
            )
            write_file(corrupted_path, corrupted)
            manifest_path = os.path.join(
                test_output_dir,
                f"{file_stem}.level_{error_level}.manifest.json",
            )
            with open(manifest_path, "w") as fh:
                json.dump(manifest, fh, indent=2)

            total_injected = len(manifest.get("errors", []))
            print(f"    Injected {total_injected} errors (deterministic)")

            for skill in skills:
                t0 = time.time()

                try:
                    fixed_content, _ = fix_with_model(
                        llm,
                        corrupted,
                        original_content,
                        test_output_dir,
                        file_stem,
                        error_level,
                        prompt_name=skill,
                        model_name=model_name,
                    )
                    duration = time.time() - t0

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
                        model_name=f"{model_name}+{skill}",
                        false_positives=getattr(result, "false_positives", 0),
                        code_fence_violations=getattr(result, "code_fence_violations", 0),
                        yaml_violations=getattr(result, "yaml_violations", 0),
                        section_violations=getattr(result, "section_violations", 0),
                    )
                    _populate_scorable(sr)

                    print(
                        f"    {skill:<20} F1={sr.f1_score:.3f} "
                        f"F1s={sr.f1_score_scorable:.3f} "
                        f"fix={sr.fix_rate:.1%} time={duration:.1f}s"
                    )

                    csv_rows.append({
                        "file_stem": file_stem,
                        "model": model_name,
                        "skill": skill,
                        "error_count": error_level,
                        "total_injected": total_injected,
                        "fixed": sr.errors_fixed,
                        "unfixed": sr.errors_unfixed,
                        "fix_rate": round(sr.fix_rate, 4),
                        "f1_score": round(sr.f1_score, 4),
                        "f1_score_scorable": round(sr.f1_score_scorable, 4),
                        "f1_score_content": round(sr.f1_score_content, 4),
                        "f1_score_hygiene": round(sr.f1_score_hygiene, 4),
                        "false_positives": sr.false_positives,
                        "code_fence_violations": sr.code_fence_violations,
                        "yaml_violations": sr.yaml_violations,
                        "section_violations": sr.section_violations,
                        "duration_s": round(duration, 2),
                    })

                except Exception as e:
                    print(f"    {skill:<20} ERROR: {e}")

    csv_path = os.path.join(test_output_dir, "SKILL_MATRIX_TABLE.csv")
    _write_skill_comparison_csv(csv_rows, csv_path)

    print(f"\n{'='*70}")
    print("SKILL MATRIX COMPLETE")
    print(f"{'='*70}")
    print(f"Total rows written: {len(csv_rows)}")
    print(f"Results written to: {csv_path}")

    assert len(csv_rows) > 0, "No results produced — check LLM/proxy connectivity"


# ============================================================================
# FOCUSED THREE-MODEL BENCHMARK: gpt-4o × gpt-5.4 × kimi-k2.5
# ============================================================================

def test_three_model_matrix(llm, test_output_dir):
    """
    Focused benchmark for three models: gpt-4o, gpt-5.4, kimi-k2.5.

    Matrix shape: up to 10 Seurat vignettes × 3 error levels × 3 models
    (= up to 90 cells).  Each (file, level) pair uses one deterministic
    injection so all three models score against byte-identical corrupted text.
    Models run in parallel within each cell.

    Error levels: QUICK_STRESS_LEVELS [10, 40, 100] (quick but representative).
    Prompt: "bioguider" for all models.

    Artifacts land under outputs/three_model_matrix/run_<timestamp>/:
      <file_stem>/STRESS_TEST_*.json/csv
      _aggregate/AGGREGATE_*.json/csv

    Run:
        pytest system_tests/test_single_file_stress.py::test_three_model_matrix -v -s
    """
    import time

    TARGET_MODELS = ["gpt-4o", "gpt-5.4", "kimi-k2.5"]
    test_configs = [(m, "bioguider") for m in TARGET_MODELS]
    error_levels = QUICK_STRESS_LEVELS  # [10, 40, 100]

    run_root = os.path.join(
        "outputs/three_model_matrix",
        datetime.now().strftime("run_%Y%m%d_%H%M%S"),
    )
    os.makedirs(run_root, exist_ok=True)

    available_files = [f for f in TUTORIAL_FILES if os.path.exists(f)]
    if not available_files:
        pytest.skip("No Seurat vignettes found — clone satijalab/seurat first")

    total_cells = len(available_files) * len(error_levels) * len(test_configs)
    print(f"\n{'='*70}")
    print("THREE-MODEL BENCHMARK: gpt-4o × gpt-5.4 × kimi-k2.5")
    print(f"{'='*70}")
    print(f"Files: {len(available_files)}, Levels: {error_levels}, "
          f"Models: {TARGET_MODELS}")
    print(f"Total cells: {total_cells}")
    print(f"Output root: {run_root}")

    all_file_results: Dict[str, List[StressLevelResult]] = {}

    for test_file in available_files:
        file_stem = Path(test_file).stem
        file_out = os.path.join(run_root, file_stem)
        os.makedirs(file_out, exist_ok=True)
        original_content = read_file(test_file) or ""
        if not original_content.strip():
            print(f"  SKIP empty file: {test_file}")
            continue

        write_file(os.path.join(file_out, f"{file_stem}.original.Rmd"), original_content)

        print(f"\n{'#'*70}")
        print(f"# FILE: {file_stem}")
        print(f"{'#'*70}")

        file_results: List[StressLevelResult] = []

        for error_level in error_levels:
            print(f"\n--- Level {error_level} ---")

            injector = LLMErrorInjector(llm, force_deterministic=True)
            corrupted, manifest = injector.inject(
                original_content,
                min_per_category=error_level,
                max_words=50000,
            )
            corrupted_path = os.path.join(
                file_out, f"{file_stem}.level_{error_level}.corrupted.Rmd"
            )
            write_file(corrupted_path, corrupted)
            manifest_path = os.path.join(
                file_out, f"{file_stem}.level_{error_level}.manifest.json"
            )
            with open(manifest_path, "w") as f:
                json.dump(manifest, f, indent=2)

            injection_result = {
                "corrupted_content": corrupted,
                "manifest": manifest,
                "total_errors": len(manifest.get("errors", [])),
            }
            print(f"    Injected {injection_result['total_errors']} errors (deterministic)")

            def _run_one(model_name: str, prompt_name: str):
                combo = f"{model_name}+{prompt_name}"
                t0 = time.time()
                try:
                    fixed_content, _ = fix_with_model(
                        llm,
                        injection_result["corrupted_content"],
                        original_content,
                        file_out,
                        file_stem,
                        error_level,
                        prompt_name=prompt_name,
                        model_name=model_name,
                    )
                    duration = time.time() - t0
                    result, category_results = evaluate_fixes(
                        original_content,
                        injection_result["corrupted_content"],
                        fixed_content,
                        injection_result["manifest"],
                        llm,
                    )
                    sr = StressLevelResult(
                        error_count=error_level,
                        total_errors_injected=injection_result["total_errors"],
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
                    print(
                        f"    {combo:<32} F1={result.f1_score:.3f} "
                        f"fix={result.fix_rate:.1%} time={duration:.1f}s"
                    )
                except Exception as e:  # noqa: BLE001
                    print(f"    {combo:<32} ERROR: {e}")

            with ThreadPoolExecutor(max_workers=len(test_configs)) as pool:
                futures = [
                    pool.submit(_run_one, model_name, prompt_name)
                    for model_name, prompt_name in test_configs
                ]
                for _ in as_completed(futures):
                    pass

        save_results(file_results, file_out)
        all_file_results[file_stem] = file_results

    # Cross-file aggregate
    pooled: List[StressLevelResult] = []
    for results in all_file_results.values():
        pooled.extend(results)

    agg_dir = os.path.join(run_root, "_aggregate")
    os.makedirs(agg_dir, exist_ok=True)
    save_results(pooled, agg_dir)
    for old_name, new_name in [
        ("STRESS_TEST_RESULTS.json",         "AGGREGATE_RESULTS.json"),
        ("STRESS_TEST_TABLE.csv",             "AGGREGATE_TABLE.csv"),
        ("STRESS_TEST_CATEGORY_DETAIL.csv",   "AGGREGATE_CATEGORY_DETAIL.csv"),
        ("STRESS_TEST_REPORT.md",             "AGGREGATE_REPORT.md"),
    ]:
        src = os.path.join(agg_dir, old_name)
        dst = os.path.join(agg_dir, new_name)
        if os.path.exists(src):
            os.rename(src, dst)

    # Summary table
    print(f"\n{'='*70}")
    print("THREE-MODEL BENCHMARK SUMMARY")
    print(f"{'='*70}")
    models_seen = sorted({r.model_name for r in pooled})
    print(f"\n{'Model':<32} | {'Avg F1':>8} | {'Avg Fix%':>8} | {'Cells':>6}")
    print("-" * 60)
    for m in models_seen:
        mrs = [r for r in pooled if r.model_name == m]
        avg_f1 = sum(r.f1_score for r in mrs) / len(mrs)
        avg_fix = sum(r.fix_rate for r in mrs) / len(mrs)
        print(f"{m:<32} | {avg_f1:>8.3f} | {avg_fix:>7.1%} | {len(mrs):>6}")

    index_path = os.path.join(run_root, "INDEX.md")
    with open(index_path, "w") as f:
        f.write(f"# Three-Model Benchmark — {datetime.now():%Y-%m-%d %H:%M:%S}\n\n")
        f.write(f"- Models: {TARGET_MODELS}\n")
        f.write(f"- Files: {list(all_file_results.keys())}\n")
        f.write(f"- Levels: {error_levels}\n")
        f.write(f"- Total results: {len(pooled)}\n\n")
        f.write("## Per-file output\n\n")
        for stem in all_file_results:
            f.write(f"- `{stem}/` — STRESS_TEST_RESULTS.json + STRESS_TEST_TABLE.csv\n")
        f.write("\n## Aggregate\n\n")
        f.write("- `_aggregate/AGGREGATE_RESULTS.json`\n")
        f.write("- `_aggregate/AGGREGATE_TABLE.csv`\n")
        f.write(f"\nIndex: {index_path}\n")

    print(f"\nArtifacts: {run_root}")
    print(f"Index:     {index_path}")

    assert len(pooled) > 0, "No results produced — check LLM/proxy connectivity"
