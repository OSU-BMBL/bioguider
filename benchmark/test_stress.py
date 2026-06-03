"""
Original single-file stress tests.
"""
import os
from pathlib import Path

import pytest

from benchmark.shared import *


def test_single_file_stress(llm, test_output_dir):
    """
    Run stress test on a single file with multiple error levels.

    This is the main test - runs all stress levels in parallel.
    """
    test_file = DEFAULT_TEST_FILE

    # Verify file exists
    if not os.path.exists(test_file):
        pytest.skip(f"Test file not found: {test_file}")

    results = run_stress_test_parallel(
        llm=llm,
        test_file=test_file,
        stress_levels=STRESS_LEVELS,
        output_dir=test_output_dir,
        max_workers=MAX_WORKERS
    )

    # Save results
    save_results(results, test_output_dir)

    # Print summary
    print("\n" + "=" * 60)
    print("STRESS TEST SUMMARY")
    print("=" * 60)
    print(f"{'Errors':>8} | {'Fixed':>6} | {'Unfixed':>7} | {'F1':>6} | {'Fix Rate':>8}")
    print("-" * 60)
    for r in results:
        print(f"{r.error_count:>8} | {r.errors_fixed:>6} | {r.errors_unfixed:>7} | "
              f"{r.f1_score:>6.3f} | {r.fix_rate:>8.1%}")
    print("=" * 60)

    # Assertions
    assert len(results) == len(STRESS_LEVELS), "Should have results for all levels"
    for r in results:
        assert r.total_errors_injected > 0, f"Should inject errors at level {r.error_count}"


def test_single_file_stress_quick(llm, test_output_dir):
    """
    Quick stress test with fewer levels for faster iteration.
    """
    test_file = DEFAULT_TEST_FILE

    if not os.path.exists(test_file):
        pytest.skip(f"Test file not found: {test_file}")

    results = run_stress_test_parallel(
        llm=llm,
        test_file=test_file,
        stress_levels=QUICK_STRESS_LEVELS,
        output_dir=test_output_dir,
        max_workers=MAX_WORKERS
    )

    save_results(results, test_output_dir)

    print("\nQuick stress test complete:")
    for r in results:
        print(f"  Level {r.error_count}: F1={r.f1_score:.3f}, FixRate={r.fix_rate:.1%}")

    assert len(results) == len(QUICK_STRESS_LEVELS)


def test_single_file_stress_minimal(llm, test_output_dir):
    """
    Minimal test with just one level - for pipeline verification.
    """
    test_file = DEFAULT_TEST_FILE

    if not os.path.exists(test_file):
        pytest.skip(f"Test file not found: {test_file}")

    results = run_stress_test_parallel(
        llm=llm,
        test_file=test_file,
        stress_levels=[10],  # Single level
        output_dir=test_output_dir,
        max_workers=1  # Single worker for debugging
    )

    save_results(results, test_output_dir)

    assert len(results) == 1
    r = results[0]
    print(f"\nMinimal test: {r.total_errors_injected} errors injected, {r.errors_fixed} fixed (F1={r.f1_score:.3f})")


def test_prepare_for_other_models(llm, test_output_dir):
    """
    Prepare corrupted files for testing with other models (GPT-5.1, Claude, Gemini).

    This creates corrupted files that you can manually fix with each model.
    """
    test_file = DEFAULT_TEST_FILE

    if not os.path.exists(test_file):
        pytest.skip(f"Test file not found: {test_file}")

    original_content = read_file(test_file)
    file_basename = Path(test_file).stem

    # Create directories for each model
    models = ["bioguider", "gpt-5.1", "claude-sonnet", "gemini"]
    for model in models:
        model_dir = os.path.join(test_output_dir, f"fixed_{model}")
        os.makedirs(model_dir, exist_ok=True)

    # Inject errors at a medium level (20)
    injection_result = inject_errors_at_level(
        llm, original_content, 20, test_output_dir, file_basename
    )

    # Create instructions file
    instructions = f"""# Model Comparison Instructions

## Test File
- Original: `{file_basename}.original.Rmd`
- Corrupted: `{file_basename}.level_20.corrupted.Rmd`
- Errors injected: {injection_result['total_errors']}

## Steps for Each Model
1. Open the corrupted file in Cursor
2. Set the AI model (GPT-5.1, Claude Sonnet, or Gemini)
3. Prompt: "Fix all errors, typos, and formatting issues in this file"
4. Save the fixed file to `fixed_{{model_name}}/{file_basename}.fixed.Rmd`

## After Fixing
Run the evaluation:
```
pytest system_tests/test_single_file_stress.py::test_evaluate_model_comparison -v -s
```
"""

    instructions_path = os.path.join(test_output_dir, "INSTRUCTIONS.md")
    with open(instructions_path, 'w') as f:
        f.write(instructions)

    print("\nPrepared files for model comparison:")
    print(f"  - Corrupted file: {injection_result['corrupted_path']}")
    print(f"  - Errors: {injection_result['total_errors']}")
    print(f"  - Instructions: {instructions_path}")
    print("\nFix with each model and save to fixed_{model}/ directories")


def test_multi_model_comparison(llm, test_output_dir):
    """
    Compare multiple models and prompts on the same corrupted document.

    Tests:
    - GPT-4o with BioGuider prompt (domain-specific)
    - GPT-4o with simple prompt (baseline)
    - GPT-4o with basic prompt (minimal)
    - Ollama models (qwen, gpt-oss) with simple prompt

    Uses only 10 errors for quick testing.
    """
    test_file = DEFAULT_TEST_FILE

    if not os.path.exists(test_file):
        pytest.skip(f"Test file not found: {test_file}")

    original_content = read_file(test_file)
    file_basename = Path(test_file).stem

    # Use only 10 errors for quick testing
    error_level = 10

    # Print available prompts and models
    print_prompts()
    print_models()

    print(f"{'='*70}")
    print("MULTI-MODEL COMPARISON TEST (10 errors)")
    print(f"{'='*70}")
    print(f"File: {test_file}")
    print(f"Error level: {error_level}")

    # Inject errors once
    print(f"\nInjecting {error_level} errors per category...")
    injection_result = inject_errors_at_level(
        llm, original_content, error_level, test_output_dir, file_basename
    )
    print(f"Total injected: {injection_result['total_errors']} errors")

    # Save original and corrupted
    write_file(os.path.join(test_output_dir, f"{file_basename}.original.Rmd"), original_content)

    all_results = []

    # Define test configurations: (model_name, prompt_name)
    test_configs = [
        ("gpt-4o", "bioguider"),      # GPT-4o with BioGuider prompt (should be best)
        ("gpt-4o", "simple"),         # GPT-4o with simple prompt
        ("claude_sonnet", "simple"), # Claude Sonnet with simple prompt
        ("qwen3_30b", "simple"),     # Qwen 30B (balanced) with simple
        ("gpt_oss_20b", "simple"),   # GPT-OSS 20B with simple
        ("qwen3_0.6b", "simple"),    # Qwen 0.6B (fast, small) with simple
    ]

    for model_name, prompt_name in test_configs:
        model_desc = MODELS.get(model_name, {}).get("description", model_name)
        prompt_desc = PROMPTS.get(prompt_name, {}).get("description", prompt_name)[:40]

        print(f"\n--- Testing: {model_name} + {prompt_name} ---")
        print(f"    Model: {model_desc}")
        print(f"    Prompt: {prompt_desc}...")

        try:
            fixed_content, _ = fix_with_model(
                llm,
                injection_result["corrupted_content"],
                original_content,
                test_output_dir,
                file_basename,
                error_level,
                prompt_name=prompt_name,
                model_name=model_name
            )

            # Evaluate
            result, category_results = evaluate_fixes(
                original_content,
                injection_result["corrupted_content"],
                fixed_content,
                injection_result["manifest"],
                llm
            )

            combo_name = f"{model_name}+{prompt_name}"
            stress_result = StressLevelResult(
                error_count=error_level,
                total_errors_injected=injection_result["total_errors"],
                errors_fixed=result.true_positives,
                errors_unfixed=result.false_negatives,
                fix_rate=result.fix_rate,
                precision=result.precision,
                recall=result.recall,
                f1_score=result.f1_score,
                duration_seconds=0,
                category_results=category_results,
                model_name=combo_name
            )

            all_results.append(stress_result)

            print(f"    Result: Fixed {result.true_positives}/{injection_result['total_errors']} "
                  f"({result.fix_rate:.1%}), F1={result.f1_score:.3f}")

        except Exception as e:
            print(f"    ERROR: {e}")
            continue

    # Save comparison results
    save_results(all_results, test_output_dir)

    # Print comparison summary
    print(f"\n{'='*70}")
    print("COMPARISON SUMMARY")
    print(f"{'='*70}")
    print(f"{'Model+Prompt':<30} {'Fixed':<8} {'Rate':<8} {'F1':<8}")
    print("-" * 70)

    # Sort by F1 score
    all_results.sort(key=lambda r: r.f1_score, reverse=True)

    for r in all_results:
        print(f"{r.model_name:<30} {r.errors_fixed:<8} {r.fix_rate:.1%}{'':>2} {r.f1_score:.3f}")

    # Find best result
    if all_results:
        best = all_results[0]
        print(f"\nBest: {best.model_name} with F1={best.f1_score:.3f}")

        # Check if BioGuider is best
        bioguider_results = [r for r in all_results if "bioguider" in r.model_name]
        if bioguider_results and bioguider_results[0] == best:
            print("✓ BioGuider prompt achieved best results!")

    assert len(all_results) >= 1, "Should have at least one result"


def test_model_comparison(llm, test_output_dir):
    """
    Quick comparison: BioGuider vs simple prompt on GPT-4o only.

    Uses 10 errors for fast testing.
    """
    test_file = DEFAULT_TEST_FILE

    if not os.path.exists(test_file):
        pytest.skip(f"Test file not found: {test_file}")

    original_content = read_file(test_file)
    file_basename = Path(test_file).stem

    # Use only 10 errors for quick testing
    error_level = 10

    print(f"\n{'='*60}")
    print("QUICK MODEL COMPARISON (GPT-4o: bioguider vs simple)")
    print(f"{'='*60}")
    print(f"File: {test_file}")
    print(f"Error level: {error_level}")

    # Inject errors once
    print("\nInjecting errors...")
    injection_result = inject_errors_at_level(
        llm, original_content, error_level, test_output_dir, file_basename
    )
    print(f"Injected {injection_result['total_errors']} errors")

    # Save original
    write_file(os.path.join(test_output_dir, f"{file_basename}.original.Rmd"), original_content)

    all_results = []

    # Test both prompts with GPT-4o
    for prompt_name in ["bioguider", "simple"]:
        print(f"\n--- Testing GPT-4o with {prompt_name} prompt ---")

        fixed_content, _ = fix_with_model(
            llm,
            injection_result["corrupted_content"],
            original_content,
            test_output_dir,
            file_basename,
            error_level,
            prompt_name=prompt_name,
            model_name="gpt-4o"
        )

        # Evaluate
        result, category_results = evaluate_fixes(
            original_content,
            injection_result["corrupted_content"],
            fixed_content,
            injection_result["manifest"],
            llm
        )

        combo_name = f"gpt-4o+{prompt_name}"
        stress_result = StressLevelResult(
            error_count=error_level,
            total_errors_injected=injection_result["total_errors"],
            errors_fixed=result.true_positives,
            errors_unfixed=result.false_negatives,
            fix_rate=result.fix_rate,
            precision=result.precision,
            recall=result.recall,
            f1_score=result.f1_score,
            duration_seconds=0,
            category_results=category_results,
            model_name=combo_name
        )

        all_results.append(stress_result)

        print(f"  {combo_name}: Fixed {result.true_positives}/{injection_result['total_errors']} "
              f"({result.fix_rate:.1%}), F1={result.f1_score:.3f}")

    # Save comparison results
    save_results(all_results, test_output_dir)

    # Print comparison summary
    print(f"\n{'='*60}")
    print("COMPARISON SUMMARY")
    print(f"{'='*60}")
    print(f"{'Model+Prompt':<25} {'Fixed':<10} {'Fix Rate':<12} {'F1 Score':<10}")
    print("-" * 60)
    for r in all_results:
        print(f"{r.model_name:<25} {r.errors_fixed:<10} {r.fix_rate:.1%}{'':>5} {r.f1_score:.3f}")

    # Calculate difference
    bioguider_result = next((r for r in all_results if "bioguider" in r.model_name), None)
    simple_result = next((r for r in all_results if "simple" in r.model_name), None)

    if bioguider_result and simple_result:
        diff = bioguider_result.f1_score - simple_result.f1_score
        if diff > 0:
            print(f"\nBioGuider advantage: +{diff:.3f} F1 ({diff/simple_result.f1_score*100:.1f}% better)")
        elif diff < 0:
            print(f"\nSimple prompt advantage: +{-diff:.3f} F1 ({-diff/bioguider_result.f1_score*100:.1f}% better)")
        else:
            print(f"\nBoth prompts performed equally (F1={bioguider_result.f1_score:.3f})")

    assert len(all_results) == 2, "Should have results for both prompts"


def test_e004_pipeline_vs_prompt(llm, test_output_dir):
    """
    E004: Does the BioGuider evaluate-then-generate pipeline beat a direct
    LLM call with the BioGuider structured prompt or a generic one-liner?

    Injects errors once into de_vignette.Rmd (same file as E001–E003), then
    runs three strategies against the identical corrupted document for each
    model in TARGET_MODELS:
      - <model>+bioguider : direct LLM call with BioGuider's structured prompt (E001/E002)
      - <model>+simple    : direct LLM call with a one-line generic prompt (E002 baseline)
      - <model>+pipeline  : DocumentPipeline evaluate→generate, eval_type=TUTORIAL (E004)

    Because all strategies receive the same injected doc, F1 differences are
    attributable solely to the fixing strategy, not injection randomness.

    Environment variables:
      E004_MODELS      Comma-separated model keys (default: gpt-4o,gpt-5.4,kimi-k2.5,glm-5,gpt-oss)
      E004_ERROR_LEVEL Errors per category to inject (default: 10)

    Run:
        pytest benchmark/test_stress.py::test_e004_pipeline_vs_prompt -v -s
        E004_MODELS=gpt-4o,gpt-oss E004_ERROR_LEVEL=40 pytest benchmark/test_stress.py::test_e004_pipeline_vs_prompt -v -s
    """
    import time
    from concurrent.futures import ThreadPoolExecutor, as_completed as _as_completed
    from bioguider.generation.document_pipeline import DocumentPipeline
    from bioguider.utils.constants import EvaluationTypeEnum
    from langchain_openai import ChatOpenAI

    _default_models = "gpt-4o,gpt-5.4,kimi-k2.5,glm-5,gpt-oss"
    TARGET_MODELS = [m.strip() for m in os.environ.get("E004_MODELS", _default_models).split(",") if m.strip()]
    ERROR_LEVEL = int(os.environ.get("E004_ERROR_LEVEL", "10"))
    test_file = DEFAULT_TEST_FILE

    if not os.path.exists(test_file):
        pytest.skip(f"Test file not found: {test_file}")

    original_content = read_file(test_file)
    file_basename = Path(test_file).stem

    print(f"\n{'='*70}")
    print("E004: BIOGUIDER PIPELINE vs BIOGUIDER PROMPT vs SIMPLE PROMPT")
    print(f"{'='*70}")
    print(f"File       : {test_file}")
    print(f"Models     : {TARGET_MODELS}")
    print(f"Error level: {ERROR_LEVEL}")

    # ── Inject errors once — shared across all strategies ────────────────────
    injection_result = inject_errors_at_level(
        llm, original_content, ERROR_LEVEL, test_output_dir, file_basename
    )
    total_injected = injection_result["total_errors"]
    corrupted_content = injection_result["corrupted_content"]
    corrupted_filename = f"{file_basename}.level_{ERROR_LEVEL}.corrupted.Rmd"
    manifest = injection_result["manifest"]
    print(f"Injected   : {total_injected} errors")
    write_file(os.path.join(test_output_dir, f"{file_basename}.original.Rmd"), original_content)

    # ── Build shared DocumentPipeline once (LLM-independent prep step) ───────
    print(f"\nPreparing DocumentPipeline from {SEURAT_REPO_PATH} ...")
    shared_pipeline = DocumentPipeline(SEURAT_REPO_PATH).prepare_repo(llm)
    print("DocumentPipeline ready.")

    all_results: List[StressLevelResult] = []

    # ── Strategy runners ─────────────────────────────────────────────────────

    def _run_prompt(model_name: str, prompt_name: str) -> "StressLevelResult | None":
        combo = f"{model_name}+{prompt_name}"
        t0 = time.time()
        try:
            fixed_content, _ = fix_with_model(
                llm,
                corrupted_content,
                original_content,
                test_output_dir,
                file_basename,
                ERROR_LEVEL,
                prompt_name=prompt_name,
                model_name=model_name,
            )
            duration = time.time() - t0
            result, category_results = evaluate_fixes(
                original_content, corrupted_content, fixed_content, manifest, llm
            )
            sr = StressLevelResult(
                error_count=ERROR_LEVEL,
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
            )
            print(f"  {combo:<42} F1={result.f1_score:.3f} fix={result.fix_rate:.1%} time={duration:.1f}s")
            return sr
        except Exception as e:
            print(f"  {combo:<42} ERROR: {e}")
            return None

    def _run_pipeline(model_name: str) -> "StressLevelResult | None":
        combo = f"{model_name}+pipeline"
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
                proxy_key, proxy_base_url = resolve_proxy_credentials()
                model_llm = ChatOpenAI(
                    model=model_id,
                    api_key=proxy_key,
                    base_url=proxy_base_url,
                    timeout=120,
                    max_retries=1,
                )
            report_path = os.path.join(
                test_output_dir,
                f"{file_basename}.level_{ERROR_LEVEL}.{model_name}.pipeline_report.json",
            )
            _, fixed_content = shared_pipeline.evaluate_and_refine_document(
                llm=model_llm,
                doc_repo_path=test_output_dir,
                doc_path=corrupted_filename,
                eval_type=EvaluationTypeEnum.TUTORIAL,
                report_output_path=report_path,
            )
            duration = time.time() - t0
            write_file(
                os.path.join(
                    test_output_dir,
                    f"{file_basename}.level_{ERROR_LEVEL}.{model_name}.pipeline_fixed.Rmd",
                ),
                fixed_content,
            )
            result, category_results = evaluate_fixes(
                original_content, corrupted_content, fixed_content, manifest, llm
            )
            sr = StressLevelResult(
                error_count=ERROR_LEVEL,
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
            )
            print(f"  {combo:<42} F1={result.f1_score:.3f} fix={result.fix_rate:.1%} time={duration:.1f}s")
            return sr
        except Exception as e:
            print(f"  {combo:<42} ERROR: {e}")
            return None

    # ── Run 3 strategies × N models in parallel ───────────────────────────────
    n_strategies = 3  # bioguider, simple, pipeline
    print(f"\nRunning {len(TARGET_MODELS) * n_strategies} tasks in parallel ...")
    with ThreadPoolExecutor(max_workers=len(TARGET_MODELS) * n_strategies) as pool:
        futures = {}
        for model_name in TARGET_MODELS:
            futures[pool.submit(_run_prompt, model_name, "bioguider")] = f"{model_name}+bioguider"
            futures[pool.submit(_run_prompt, model_name, "simple")] = f"{model_name}+simple"
            futures[pool.submit(_run_pipeline, model_name)] = f"{model_name}+pipeline"
        for future in _as_completed(futures):
            sr = future.result()
            if sr is not None:
                all_results.append(sr)

    save_results(all_results, test_output_dir)

    # ── Comparison table ──────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"E004 COMPARISON SUMMARY  (level={ERROR_LEVEL}, file={file_basename})")
    print(f"{'='*70}")
    print(f"{'Strategy':<42} | {'F1':>6} | {'Fix%':>6} | {'Prec':>6} | {'Time':>7}")
    print("-" * 72)
    for sr in sorted(all_results, key=lambda r: -r.f1_score):
        print(
            f"{sr.model_name:<42} | {sr.f1_score:>6.3f} | {sr.fix_rate:>5.1%} "
            f"| {sr.precision:>6.3f} | {sr.duration_seconds:>6.1f}s"
        )

    # Per-model pipeline vs bioguider-prompt delta
    print(f"\n--- Pipeline vs BioGuider-prompt delta (per model) ---")
    for model_name in TARGET_MODELS:
        bio_r  = next((r for r in all_results if r.model_name == f"{model_name}+bioguider"), None)
        pipe_r = next((r for r in all_results if r.model_name == f"{model_name}+pipeline"), None)
        if bio_r and pipe_r:
            delta = pipe_r.f1_score - bio_r.f1_score
            print(f"  {model_name}: pipeline {delta:+.3f} F1 vs bioguider prompt")

    assert all_results, "No results produced — check LLM/proxy connectivity"


def test_full_benchmark(llm, test_output_dir):
    """
    FULL BENCHMARK - Two parts:
    1. Model comparison: All models at 30 errors
    2. Stress test: BioGuider only from 10 to 300 errors

    Not yet implemented — superseded by test_all_models_all_levels.
    """
    pytest.skip("Not implemented — use test_all_models_all_levels instead")


def test_all_models_all_levels(llm, test_output_dir):
    """
    COMPREHENSIVE BENCHMARK: All models x All error levels.

    Tests all models at error levels: 10, 30, 50, 100, 200, 300
    """
    test_file = DEFAULT_TEST_FILE

    if not os.path.exists(test_file):
        pytest.skip(f"Test file not found: {test_file}")

    original_content = read_file(test_file)
    file_basename = Path(test_file).stem

    # Error levels to test
    error_levels = [10, 30, 50, 100, 200, 300]

    # All models with bioguider prompt + one simple-prompt baseline (AC6: ≥4 series)
    test_configs = [(m, "bioguider") for m in MODELS] + [("gpt-5.4", "simple")]

    print_prompts()
    print_models()

    print(f"\n{'='*70}")
    print("COMPREHENSIVE BENCHMARK: ALL MODELS x ALL ERROR LEVELS")
    print(f"{'='*70}")
    print(f"File: {test_file}")
    print(f"Error levels: {error_levels}")
    print(f"Models: {[f'{m}+{p}' for m, p in test_configs]}")
    print(f"Total tests: {len(error_levels) * len(test_configs)}")

    # Save original
    write_file(os.path.join(test_output_dir, f"{file_basename}.original.Rmd"), original_content)

    all_results = []
    import time

    # For each error level
    for error_level in error_levels:
        print(f"\n{'='*70}")
        print(f"ERROR LEVEL: {error_level}")
        print(f"{'='*70}")

        # Inject errors for this level
        print(f"Injecting {error_level} errors per category...")
        injection_result = inject_errors_at_level(
            llm, original_content, error_level, test_output_dir, file_basename
        )
        print(f"Total injected: {injection_result['total_errors']} errors")

        # Test all models at this level
        for model_name, prompt_name in test_configs:
            combo_name = f"{model_name}+{prompt_name}"
            print(f"\n--- {combo_name} @ Level {error_level} ---")

            try:
                start_time = time.time()

                fixed_content, _ = fix_with_model(
                    llm,
                    injection_result["corrupted_content"],
                    original_content,
                    test_output_dir,
                    file_basename,
                    error_level,
                    prompt_name=prompt_name,
                    model_name=model_name
                )

                duration = time.time() - start_time

                result, category_results = evaluate_fixes(
                    original_content,
                    injection_result["corrupted_content"],
                    fixed_content,
                    injection_result["manifest"],
                    llm
                )

                stress_result = StressLevelResult(
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
                    model_name=combo_name
                )

                all_results.append(stress_result)
                print(f"    Fixed {result.true_positives}/{injection_result['total_errors']} "
                      f"({result.fix_rate:.1%}), F1={result.f1_score:.3f}, Time={duration:.1f}s")

            except Exception as e:
                print(f"    ERROR: {e}")
                continue

        # Save after each level
        save_results(all_results, test_output_dir)

    # Final summary
    print(f"\n{'='*70}")
    print("COMPREHENSIVE BENCHMARK SUMMARY")
    print(f"{'='*70}")

    # Create summary by model
    models = list(set(r.model_name for r in all_results))
    models.sort()

    print("\n--- AVERAGE PERFORMANCE BY MODEL ---")
    print(f"{'Model':<25} | {'Avg F1':>8} | {'Avg Fix%':>8} | {'Tests':>6}")
    print("-" * 55)

    model_avg = {}
    for model in models:
        model_results = [r for r in all_results if r.model_name == model]
        avg_f1 = sum(r.f1_score for r in model_results) / len(model_results)
        avg_fix = sum(r.fix_rate for r in model_results) / len(model_results)
        model_avg[model] = avg_f1
        print(f"{model:<25} | {avg_f1:>8.3f} | {avg_fix:>7.1%} | {len(model_results):>6}")

    # Find best model
    best_model = max(model_avg, key=model_avg.get)
    print(f"\n✓ Best overall: {best_model} (Avg F1={model_avg[best_model]:.3f})")

    # Create pivot table by level
    print("\n--- F1 SCORE BY MODEL AND ERROR LEVEL ---")
    header = f"{'Model':<25} |"
    for level in error_levels:
        header += f" {level:>6} |"
    print(header)
    print("-" * (28 + 9 * len(error_levels)))

    for model in models:
        row = f"{model:<25} |"
        for level in error_levels:
            result = next((r for r in all_results if r.model_name == model and r.error_count == level), None)
            if result:
                row += f" {result.f1_score:>6.3f} |"
            else:
                row += f" {'N/A':>6} |"
        print(row)

    assert len(all_results) >= len(test_configs), "Should have results for all models"
