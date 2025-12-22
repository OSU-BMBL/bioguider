"""
Comprehensive Benchmark Tests for Error Injection Module.

This test suite provides:
1. Stress testing (10, 20, 40, 60, 100 errors) with detection capability tracking
2. Multi-model comparison support (BioGuider, GPT-5.1, Claude Sonnet, Gemini)
3. F-score metrics with semantic False Positive detection
4. Multi-process parallel execution for efficiency

Usage:
    # Run stress test only
    pytest system_tests/test_comprehensive_benchmark.py::test_stress_test -v -s

    # Prepare model comparison files
    pytest system_tests/test_comprehensive_benchmark.py::test_prepare_model_comparison -v -s

    # Evaluate model comparison (after manual model runs)
    pytest system_tests/test_comprehensive_benchmark.py::test_evaluate_model_comparison -v -s

    # Run full benchmark suite
    pytest system_tests/test_comprehensive_benchmark.py -v -s
"""
import os
import json
import pytest
from pathlib import Path
from datetime import datetime

from bioguider.managers.benchmark_manager import (
    BenchmarkManager,
    DEFAULT_STRESS_LEVELS,
    SUPPORTED_MODELS,
)


# ============================================================================
# CONFIGURATION
# ============================================================================

# Default test repository (Seurat)
DEFAULT_REPORT_PATH = "logs/evaluation_report_github_satijalab_seurat.json"
DEFAULT_BASELINE_REPO = "data/.adalflow/repos/satijalab_seurat"
DEFAULT_OUTPUT_PATH = "outputs"

# Stress test levels: [10, 20, 40, 60, 100]
STRESS_LEVELS = [10, 20, 40, 60, 100]

# Quick stress test levels for faster iteration
QUICK_STRESS_LEVELS = [10, 40]

# Maximum files per category for faster testing (default: 3 for speed)
MAX_FILES_PER_CATEGORY = 3

# Quick test uses even fewer files
QUICK_MAX_FILES = 2


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def benchmark_manager(llm, step_callback):
    """Create a BenchmarkManager instance."""
    return BenchmarkManager(
        llm=llm,
        step_callback=step_callback,
        max_workers=4
    )


@pytest.fixture
def test_config():
    """Get test configuration."""
    return {
        "report_path": DEFAULT_REPORT_PATH,
        "baseline_repo_path": DEFAULT_BASELINE_REPO,
        "output_base_path": DEFAULT_OUTPUT_PATH,
        "stress_levels": STRESS_LEVELS,
        "max_files_per_category": MAX_FILES_PER_CATEGORY,
    }


# ============================================================================
# STRESS TESTS
# ============================================================================

def test_stress_test(benchmark_manager, test_config):
    """
    Run stress test across multiple error levels.
    
    This test injects varying numbers of errors (10, 20, 40, 60, 100) and 
    measures BioGuider's ability to detect and fix them.
    
    NOTE: By default, only injected files are processed (not all 62 files in the
    evaluation report). This significantly speeds up testing.
    
    Output:
    - outputs/benchmark_{timestamp}/STRESS_TEST_RESULTS.json
    - outputs/benchmark_{timestamp}/STRESS_TEST_TABLE.csv
    - outputs/benchmark_{timestamp}/STRESS_TEST_REPORT.md
    """
    results = benchmark_manager.run_stress_test(
        report_path=test_config["report_path"],
        baseline_repo_path=test_config["baseline_repo_path"],
        output_base_path=test_config["output_base_path"],
        stress_levels=test_config["stress_levels"],
        max_files_per_category=test_config["max_files_per_category"],
        detect_semantic_fp=True,
        limit_generation_files=True,  # Only process injected files for speed
    )
    
    # Assertions
    assert results is not None, "Stress test should return results"
    assert len(results) == len(test_config["stress_levels"]), "Should have results for all levels"
    
    # Print summary
    print("\n" + "=" * 60)
    print("STRESS TEST SUMMARY")
    print("=" * 60)
    print(f"{'Errors':>8} | {'TP':>5} | {'FN':>5} | {'FP':>5} | {'Precision':>9} | {'Recall':>7} | {'F1':>6} | {'Fix Rate':>8}")
    print("-" * 60)
    
    for level, result in sorted(results.items()):
        br = result.benchmark_result
        print(f"{level:>8} | {br.true_positives:>5} | {br.false_negatives:>5} | "
              f"{br.false_positives:>5} | {br.precision:>9.3f} | {br.recall:>7.3f} | "
              f"{br.f1_score:>6.3f} | {br.fix_rate:>8.3f}")
    
    print("=" * 60)
    
    # Find performance drop-off
    prev_f1 = 1.0
    for level, result in sorted(results.items()):
        br = result.benchmark_result
        if br.f1_score < prev_f1 * 0.8:
            print(f"\n⚠️  Performance drop-off detected at {level} errors (F1 dropped from {prev_f1:.3f} to {br.f1_score:.3f})")
            break
        prev_f1 = br.f1_score
    else:
        print("\n✅ Performance stable across all tested error levels")
    
    # Verify output files
    sample_dir = list(results.values())[0].output_dir
    parent_dir = str(Path(sample_dir).parent)
    
    assert os.path.exists(os.path.join(parent_dir, "STRESS_TEST_RESULTS.json")), "JSON results should exist"
    assert os.path.exists(os.path.join(parent_dir, "STRESS_TEST_TABLE.csv")), "CSV table should exist"
    assert os.path.exists(os.path.join(parent_dir, "STRESS_TEST_REPORT.md")), "Report should exist"
    
    print(f"\n📁 Results saved to: {parent_dir}")
    return results


def test_stress_test_quick(benchmark_manager, test_config):
    """
    Quick stress test with reduced levels for faster iteration.
    Tests only [10, 40] errors with 2 files max per category.
    
    This is the recommended test for development iteration.
    """
    results = benchmark_manager.run_stress_test(
        report_path=test_config["report_path"],
        baseline_repo_path=test_config["baseline_repo_path"],
        output_base_path=test_config["output_base_path"],
        stress_levels=QUICK_STRESS_LEVELS,  # [10, 40]
        max_files_per_category=QUICK_MAX_FILES,  # 2 files
        detect_semantic_fp=False,  # Skip semantic FP for speed
        limit_generation_files=True,  # Only process injected files
    )
    
    assert len(results) == 2
    for level, result in results.items():
        assert result.benchmark_result.f1_score >= 0
        print(f"Level {level}: F1={result.benchmark_result.f1_score:.3f}")


def test_stress_test_minimal(benchmark_manager, test_config):
    """
    Minimal stress test for fastest iteration.
    Tests only 1 error level with 1 file per category.
    
    Use this to verify the pipeline works before running longer tests.
    """
    results = benchmark_manager.run_stress_test(
        report_path=test_config["report_path"],
        baseline_repo_path=test_config["baseline_repo_path"],
        output_base_path=test_config["output_base_path"],
        stress_levels=[10],  # Single level
        max_files_per_category=1,  # Just 1 file per category
        detect_semantic_fp=False,  # Skip semantic FP
        limit_generation_files=True,  # Only process injected files
    )
    
    assert len(results) == 1
    result = results[10]
    print(f"Minimal test: F1={result.benchmark_result.f1_score:.3f}, TP={result.benchmark_result.true_positives}, FN={result.benchmark_result.false_negatives}")


# ============================================================================
# MODEL COMPARISON TESTS
# ============================================================================

def test_prepare_model_comparison(benchmark_manager, test_config):
    """
    Prepare corrupted files for multi-model comparison.
    
    This test:
    1. Injects errors into documentation files
    2. Saves corrupted files for manual testing with other models
    3. Generates instructions for running each model
    
    After running this test:
    1. Open files in `corrupted/` with Cursor using each model
    2. Fix errors and save to `fixed_{model_name}/`
    3. Run test_evaluate_model_comparison to compute metrics
    
    Output:
    - outputs/model_comparison_{timestamp}/corrupted/
    - outputs/model_comparison_{timestamp}/originals/
    - outputs/model_comparison_{timestamp}/fixed_bioguider/
    - outputs/model_comparison_{timestamp}/fixed_gpt-5.1/
    - outputs/model_comparison_{timestamp}/fixed_claude-sonnet/
    - outputs/model_comparison_{timestamp}/fixed_gemini/
    - outputs/model_comparison_{timestamp}/INSTRUCTIONS.md
    - outputs/model_comparison_{timestamp}/BENCHMARK_MANIFEST.json
    """
    benchmark_dir = benchmark_manager.prepare_model_comparison(
        report_path=test_config["report_path"],
        baseline_repo_path=test_config["baseline_repo_path"],
        output_base_path=test_config["output_base_path"],
        error_count=20,  # Medium difficulty
        max_files_per_category=test_config["max_files_per_category"],
    )
    
    # Assertions
    assert os.path.exists(benchmark_dir), "Benchmark directory should exist"
    assert os.path.exists(os.path.join(benchmark_dir, "corrupted")), "Corrupted files should exist"
    assert os.path.exists(os.path.join(benchmark_dir, "originals")), "Original files should exist"
    assert os.path.exists(os.path.join(benchmark_dir, "BENCHMARK_MANIFEST.json")), "Manifest should exist"
    assert os.path.exists(os.path.join(benchmark_dir, "INSTRUCTIONS.md")), "Instructions should exist"
    
    # Verify model directories created
    for model in SUPPORTED_MODELS:
        model_dir = os.path.join(benchmark_dir, f"fixed_{model}")
        assert os.path.exists(model_dir), f"Directory for {model} should exist"
    
    # Print instructions
    print("\n" + "=" * 60)
    print("MODEL COMPARISON PREPARED")
    print("=" * 60)
    print(f"\n📁 Benchmark directory: {benchmark_dir}")
    print("\n📋 Next steps:")
    print(f"   1. Open files in '{benchmark_dir}/corrupted/' with Cursor")
    print("   2. For each model (GPT-5.1, Claude Sonnet, Gemini):")
    print("      - Set Cursor to use that model")
    print("      - Prompt: 'Fix all errors in this file'")
    print(f"      - Save to 'fixed_{{model_name}}/' directory")
    print("   3. Run: pytest test_comprehensive_benchmark.py::test_evaluate_model_comparison -v -s")
    print(f"      with --benchmark-dir={benchmark_dir}")
    print("=" * 60)
    
    # Save the benchmark_dir path for later use
    state_path = os.path.join(test_config["output_base_path"], ".last_benchmark_dir")
    with open(state_path, 'w') as f:
        f.write(benchmark_dir)
    
    return benchmark_dir


def test_evaluate_model_comparison(benchmark_manager, test_config):
    """
    Evaluate and compare results from multiple models.
    
    This test should be run AFTER manually fixing files with each model.
    
    Prerequisites:
    1. Run test_prepare_model_comparison first
    2. Fix files using each model and save to fixed_{model}/ directories
    3. Run this test to compute and compare metrics
    
    Output:
    - MODEL_COMPARISON_RESULTS.json
    - MODEL_COMPARISON_TABLE.csv
    - MODEL_COMPARISON_REPORT.md
    """
    # Try to load the last benchmark directory
    state_path = os.path.join(test_config["output_base_path"], ".last_benchmark_dir")
    
    if os.path.exists(state_path):
        with open(state_path, 'r') as f:
            benchmark_dir = f.read().strip()
    else:
        # Look for most recent model_comparison directory
        output_dir = Path(test_config["output_base_path"])
        comparison_dirs = sorted(
            [d for d in output_dir.iterdir() if d.is_dir() and d.name.startswith("model_comparison_")],
            key=lambda x: x.name,
            reverse=True
        )
        if not comparison_dirs:
            pytest.skip("No model comparison directory found. Run test_prepare_model_comparison first.")
        benchmark_dir = str(comparison_dirs[0])
    
    if not os.path.exists(benchmark_dir):
        pytest.skip(f"Benchmark directory not found: {benchmark_dir}")
    
    print(f"\n📁 Evaluating benchmark: {benchmark_dir}")
    
    # First, run BioGuider to fix files automatically
    manifest_path = os.path.join(benchmark_dir, "BENCHMARK_MANIFEST.json")
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)
    
    total_errors = manifest.get("total_errors", 0)
    print(f"📊 Total errors injected: {total_errors}")
    
    # Check which models have fixed files
    available_models = []
    for model in SUPPORTED_MODELS:
        model_dir = os.path.join(benchmark_dir, f"fixed_{model}")
        if os.path.exists(model_dir):
            files_in_dir = list(Path(model_dir).rglob("*"))
            file_count = sum(1 for f in files_in_dir if f.is_file())
            if file_count > 0:
                available_models.append(model)
                print(f"   ✓ {model}: {file_count} files")
            else:
                print(f"   ○ {model}: no files yet")
        else:
            print(f"   ○ {model}: directory missing")
    
    if not available_models:
        pytest.skip("No models have fixed files yet. Fix files for at least one model first.")
    
    # Evaluate all available models
    result = benchmark_manager.evaluate_model_comparison(
        benchmark_dir=benchmark_dir,
        models=available_models,
        detect_semantic_fp=True,
    )
    
    # Print comparison
    print("\n" + "=" * 60)
    print("MODEL COMPARISON RESULTS")
    print("=" * 60)
    print(f"{'Model':>15} | {'TP':>5} | {'FN':>5} | {'FP':>5} | {'Precision':>9} | {'Recall':>7} | {'F1':>6}")
    print("-" * 60)
    
    for model, br in sorted(result.results.items(), key=lambda x: -x[1].f1_score):
        print(f"{model:>15} | {br.true_positives:>5} | {br.false_negatives:>5} | "
              f"{br.false_positives:>5} | {br.precision:>9.3f} | {br.recall:>7.3f} | {br.f1_score:>6.3f}")
    
    print("=" * 60)
    
    # Rank models
    if len(result.results) > 1:
        ranked = sorted(result.results.items(), key=lambda x: x[1].f1_score, reverse=True)
        print(f"\n🏆 Best performing model: {ranked[0][0]} (F1={ranked[0][1].f1_score:.3f})")
    
    # Verify output files
    assert os.path.exists(os.path.join(benchmark_dir, "MODEL_COMPARISON_RESULTS.json")), "JSON should exist"
    assert os.path.exists(os.path.join(benchmark_dir, "MODEL_COMPARISON_TABLE.csv")), "CSV should exist"
    
    print(f"\n📁 Results saved to: {benchmark_dir}")
    return result


# ============================================================================
# F-SCORE DETAILED ANALYSIS
# ============================================================================

def test_fscore_breakdown_by_category(benchmark_manager, test_config):
    """
    Run benchmark and analyze F-score breakdown by error category.
    
    This helps identify which types of errors are hardest to fix.
    """
    results = benchmark_manager.run_stress_test(
        report_path=test_config["report_path"],
        baseline_repo_path=test_config["baseline_repo_path"],
        output_base_path=test_config["output_base_path"],
        stress_levels=[20],  # Single level for focused analysis
        max_files_per_category=test_config["max_files_per_category"],
        detect_semantic_fp=True,
    )
    
    assert 20 in results
    br = results[20].benchmark_result
    
    print("\n" + "=" * 60)
    print("F-SCORE BREAKDOWN BY CATEGORY")
    print("=" * 60)
    print(f"{'Category':>25} | {'TP':>5} | {'FN':>5} | {'Fix Rate':>8}")
    print("-" * 60)
    
    for cat, metrics in sorted(br.per_category.items(), key=lambda x: -x[1].get("tp", 0)):
        tp = metrics.get("tp", 0)
        fn = metrics.get("fn", 0)
        total = tp + fn
        fix_rate = tp / total if total > 0 else 0.0
        print(f"{cat:>25} | {tp:>5} | {fn:>5} | {fix_rate:>8.3f}")
    
    print("=" * 60)
    
    # Identify hardest categories
    hardest = sorted(
        [(cat, m) for cat, m in br.per_category.items()],
        key=lambda x: x[1].get("tp", 0) / max(1, x[1].get("tp", 0) + x[1].get("fn", 0))
    )[:3]
    
    if hardest:
        print("\n⚠️  Hardest error categories to fix:")
        for cat, metrics in hardest:
            tp = metrics.get("tp", 0)
            fn = metrics.get("fn", 0)
            rate = tp / (tp + fn) if (tp + fn) > 0 else 0
            print(f"   - {cat}: {rate:.1%} fix rate")
    
    return br


def test_fscore_breakdown_by_file(benchmark_manager, test_config):
    """
    Run benchmark and analyze F-score breakdown by file type.
    """
    results = benchmark_manager.run_stress_test(
        report_path=test_config["report_path"],
        baseline_repo_path=test_config["baseline_repo_path"],
        output_base_path=test_config["output_base_path"],
        stress_levels=[20],
        max_files_per_category=test_config["max_files_per_category"],
        detect_semantic_fp=False,  # Faster
    )
    
    br = results[20].benchmark_result
    
    print("\n" + "=" * 60)
    print("F-SCORE BREAKDOWN BY FILE")
    print("=" * 60)
    print(f"{'File':>40} | {'TP':>5} | {'FN':>5} | {'FP':>5} | {'Fix Rate':>8}")
    print("-" * 60)
    
    for fpath, metrics in sorted(br.per_file.items()):
        tp = metrics.get("tp", 0)
        fn = metrics.get("fn", 0)
        fp = metrics.get("fp", 0)
        total = tp + fn
        fix_rate = tp / total if total > 0 else 0.0
        
        # Truncate path for display
        display_path = fpath[-38:] if len(fpath) > 38 else fpath
        print(f"{display_path:>40} | {tp:>5} | {fn:>5} | {fp:>5} | {fix_rate:>8.3f}")
    
    print("=" * 60)
    return br


# ============================================================================
# SEMANTIC FALSE POSITIVE ANALYSIS
# ============================================================================

def test_semantic_fp_analysis(benchmark_manager, test_config):
    """
    Analyze semantic false positives in detail.
    
    This test uses LLM to identify harmful unintended changes.
    """
    results = benchmark_manager.run_stress_test(
        report_path=test_config["report_path"],
        baseline_repo_path=test_config["baseline_repo_path"],
        output_base_path=test_config["output_base_path"],
        stress_levels=[20],
        max_files_per_category=3,  # Small for focused analysis
        detect_semantic_fp=True,  # Enable semantic FP
    )
    
    br = results[20].benchmark_result
    
    print("\n" + "=" * 60)
    print("SEMANTIC FALSE POSITIVE ANALYSIS")
    print("=" * 60)
    print(f"\nTotal False Positives Detected: {br.false_positives}")
    
    if br.fp_details:
        print("\nDetailed FP List:")
        for i, fp in enumerate(br.fp_details, 1):
            print(f"\n{i}. File: {fp.file_path}")
            print(f"   Description: {fp.change_description}")
            print(f"   Severity: {fp.severity}")
    else:
        print("\n✅ No harmful false positives detected!")
    
    print("=" * 60)
    
    # Save FP analysis
    output_dir = str(Path(results[20].output_dir).parent)
    fp_analysis = {
        "total_fp": br.false_positives,
        "details": [
            {
                "file_path": fp.file_path,
                "description": fp.change_description,
                "severity": fp.severity,
            }
            for fp in br.fp_details
        ]
    }
    
    fp_path = os.path.join(output_dir, "SEMANTIC_FP_ANALYSIS.json")
    with open(fp_path, 'w') as f:
        json.dump(fp_analysis, f, indent=2)
    
    print(f"\n📁 FP analysis saved to: {fp_path}")
    return br


# ============================================================================
# FULL BENCHMARK SUITE
# ============================================================================

def test_full_benchmark_suite(benchmark_manager, test_config):
    """
    Run the complete benchmark suite:
    1. Stress test across all levels
    2. Prepare model comparison files
    3. Detailed F-score analysis
    
    This is the main entry point for comprehensive benchmarking.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suite_dir = os.path.join(test_config["output_base_path"], f"full_benchmark_{timestamp}")
    os.makedirs(suite_dir, exist_ok=True)
    
    print("\n" + "=" * 70)
    print("FULL BENCHMARK SUITE")
    print("=" * 70)
    
    # 1. Stress Test
    print("\n[1/3] Running Stress Test...")
    stress_results = benchmark_manager.run_stress_test(
        report_path=test_config["report_path"],
        baseline_repo_path=test_config["baseline_repo_path"],
        output_base_path=suite_dir,
        stress_levels=test_config["stress_levels"],
        max_files_per_category=test_config["max_files_per_category"],
        detect_semantic_fp=True,
    )
    
    # 2. Model Comparison Prep
    print("\n[2/3] Preparing Model Comparison...")
    comparison_dir = benchmark_manager.prepare_model_comparison(
        report_path=test_config["report_path"],
        baseline_repo_path=test_config["baseline_repo_path"],
        output_base_path=suite_dir,
        error_count=20,
        max_files_per_category=test_config["max_files_per_category"],
    )
    
    # 3. Summary Report
    print("\n[3/3] Generating Summary Report...")
    
    summary_lines = [
        "# Full Benchmark Suite Report\n",
        f"\n**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n",
        f"**Suite Directory**: {suite_dir}\n",
        "\n---\n",
        "\n## Stress Test Summary\n",
        "\n| Errors | F1 Score | Fix Rate | Duration |\n",
        "|--------|----------|----------|----------|\n",
    ]
    
    for level, result in sorted(stress_results.items()):
        br = result.benchmark_result
        summary_lines.append(
            f"| {level} | {br.f1_score:.3f} | {br.fix_rate:.3f} | {result.duration_seconds:.1f}s |\n"
        )
    
    summary_lines.extend([
        "\n---\n",
        "\n## Model Comparison\n",
        f"\nPrepared directory: `{comparison_dir}`\n",
        "\nSupported models for comparison:\n",
    ])
    
    for model in SUPPORTED_MODELS:
        summary_lines.append(f"- {model}\n")
    
    summary_lines.extend([
        "\n---\n",
        "\n## Next Steps\n",
        "\n1. Review stress test results in `STRESS_TEST_REPORT.md`\n",
        "2. Run model comparison for external models (see `INSTRUCTIONS.md`)\n",
        "3. Evaluate comparison with `test_evaluate_model_comparison`\n",
    ])
    
    summary_path = os.path.join(suite_dir, "BENCHMARK_SUITE_SUMMARY.md")
    with open(summary_path, 'w') as f:
        f.writelines(summary_lines)
    
    print("\n" + "=" * 70)
    print("BENCHMARK SUITE COMPLETE")
    print("=" * 70)
    print(f"\n📁 All results saved to: {suite_dir}")
    print(f"📄 Summary report: {summary_path}")
    print("=" * 70)
    
    return {
        "stress_results": stress_results,
        "comparison_dir": comparison_dir,
        "suite_dir": suite_dir,
    }

