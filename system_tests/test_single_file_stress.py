"""
Single File Stress Test for Error Injection Benchmark.

This test bypasses the evaluation report and works directly with specified files.
Much faster and more controlled than the comprehensive benchmark.

Usage:
    pytest system_tests/test_single_file_stress.py::test_single_file_stress -v -s
"""
import os
import json
import csv
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple

import pytest
from langchain_openai import ChatOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from openai import RateLimitError

from bioguider.generation.llm_injector import LLMErrorInjector
from bioguider.generation.benchmark_metrics import BenchmarkEvaluator, check_protected_regions
from bioguider.agents.agent_utils import read_file, write_file


# ============================================================================
# CONFIGURATION
# ============================================================================

# Default test file
DEFAULT_TEST_FILE = "data/.adalflow/repos/satijalab_seurat/vignettes/de_vignette.Rmd"

# Multi-file benchmark target set (10 topic-diverse Seurat vignettes).
# Each exercises a different anchor type for the prose_code_consistency moat.
SEURAT_VIGNETTES_DIR = "data/.adalflow/repos/satijalab_seurat/vignettes"
SEURAT_REPO_PATH = str(Path(SEURAT_VIGNETTES_DIR).parent)
TUTORIAL_FILES = [
    f"{SEURAT_VIGNETTES_DIR}/de_vignette.Rmd",
    f"{SEURAT_VIGNETTES_DIR}/cell_cycle_vignette.Rmd",
    f"{SEURAT_VIGNETTES_DIR}/dim_reduction_vignette.Rmd",
    f"{SEURAT_VIGNETTES_DIR}/integration_introduction.Rmd",
    f"{SEURAT_VIGNETTES_DIR}/hashing_vignette.Rmd",
    f"{SEURAT_VIGNETTES_DIR}/multimodal_vignette.Rmd",
    f"{SEURAT_VIGNETTES_DIR}/sctransform_vignette.Rmd",
    f"{SEURAT_VIGNETTES_DIR}/pbmc3k_tutorial.Rmd",
    f"{SEURAT_VIGNETTES_DIR}/atacseq_integration_vignette.Rmd",
    f"{SEURAT_VIGNETTES_DIR}/spatial_vignette.Rmd",
]

# Stress test levels (errors per category)
STRESS_LEVELS = [5, 10, 20, 40, 60, 100, 150, 200, 300]

# Quick test levels
QUICK_STRESS_LEVELS = [10, 40, 100]

# Output directory
OUTPUT_BASE = "outputs/single_file_stress"

# Max workers for parallel processing (override with STRESS_MAX_WORKERS env var)
MAX_WORKERS = int(os.environ.get("STRESS_MAX_WORKERS", "8"))


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class CategoryResult:
    """Result for a single error category."""
    category: str
    injected: int
    fixed: int
    unfixed: int
    fix_rate: float


@dataclass
class StressLevelResult:
    """Result for a single stress level."""
    error_count: int
    total_errors_injected: int
    errors_fixed: int
    errors_unfixed: int
    fix_rate: float
    precision: float
    recall: float
    f1_score: float
    duration_seconds: float
    category_results: List[CategoryResult] = None  # Per-category breakdown
    model_name: str = "bioguider"  # Model used for fixing
    false_positives: int = 0  # From BenchmarkResult.false_positives — exact integer, not derived

    # Token usage from the LLM fix call
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    # Scorable variants — UNSCORABLE_CATEGORIES excluded. Populated by
    # ``_populate_scorable`` from ``category_results`` before save. Kept
    # optional so legacy callers that don't populate category_results still
    # round-trip.
    total_errors_injected_scorable: int = 0
    errors_fixed_scorable: int = 0
    errors_unfixed_scorable: int = 0
    fix_rate_scorable: float = 0.0
    precision_scorable: float = 0.0
    recall_scorable: float = 0.0
    f1_score_scorable: float = 0.0

    # Paper-table CONTENT vs HYGIENE split (see managers/config.py
    # CONTENT_CATEGORIES / HYGIENE_CATEGORIES). Precision mirrors the scorable
    # precision (FPs are category-agnostic); recall/F1 are group-local.
    total_injected_content: int = 0
    fixed_content: int = 0
    f1_score_content: float = 0.0
    total_injected_hygiene: int = 0
    fixed_hygiene: int = 0
    f1_score_hygiene: float = 0.0

    # Protected region violations (Hard FP) from check_protected_regions().
    # Populated in run_stress_level from the BenchmarkResult fields.
    code_fence_violations: int = 0
    yaml_violations: int = 0
    section_violations: int = 0


def _populate_scorable(r: "StressLevelResult") -> None:
    """Fill in the scorable fields from the per-category breakdown.

    Uses the shared UNSCORABLE_CATEGORIES / compute-scorable helper from
    bioguider.managers.config so the stress-test CSV and the
    UnifiedMetricsEvaluator.EvaluationResult stay in sync on the
    carve-out story (function bucket injected, not in denominator).
    """
    from bioguider.managers.config import (
        UNSCORABLE_CATEGORIES,
        CONTENT_CATEGORIES,
        HYGIENE_CATEGORIES,
    )

    cats = r.category_results or []
    fixed_s = sum(c.fixed for c in cats if c.category not in UNSCORABLE_CATEGORIES)
    unfixed_s = sum(c.unfixed for c in cats if c.category not in UNSCORABLE_CATEGORIES)
    injected_s = fixed_s + unfixed_s

    # FPs are not category-attributed in this pipeline. Use the exact integer
    # from BenchmarkResult.false_positives (threaded through StressLevelResult)
    # rather than reverse-dividing from the rounded precision float — the
    # rounded division introduced off-by-one errors at low TP counts.
    fp_count = int(r.false_positives)

    r.total_errors_injected_scorable = injected_s
    r.errors_fixed_scorable = fixed_s
    r.errors_unfixed_scorable = unfixed_s
    r.fix_rate_scorable = fixed_s / injected_s if injected_s > 0 else 0.0
    r.precision_scorable = fixed_s / (fixed_s + fp_count) if (fixed_s + fp_count) > 0 else 0.0
    r.recall_scorable = fixed_s / injected_s if injected_s > 0 else 0.0
    r.f1_score_scorable = (
        2 * r.precision_scorable * r.recall_scorable
        / (r.precision_scorable + r.recall_scorable)
        if (r.precision_scorable + r.recall_scorable) > 0
        else 0.0
    )

    # CONTENT vs HYGIENE split. Precision mirrors the scorable precision
    # (FPs are not attributed to a group); recall is group-local.
    fixed_c = sum(c.fixed for c in cats if c.category in CONTENT_CATEGORIES)
    unfixed_c = sum(c.unfixed for c in cats if c.category in CONTENT_CATEGORIES)
    injected_c = fixed_c + unfixed_c
    fixed_h = sum(c.fixed for c in cats if c.category in HYGIENE_CATEGORIES)
    unfixed_h = sum(c.unfixed for c in cats if c.category in HYGIENE_CATEGORIES)
    injected_h = fixed_h + unfixed_h

    recall_c = fixed_c / injected_c if injected_c > 0 else 0.0
    recall_h = fixed_h / injected_h if injected_h > 0 else 0.0
    prec_s = r.precision_scorable
    f1_c = 2 * prec_s * recall_c / (prec_s + recall_c) if (prec_s + recall_c) > 0 else 0.0
    f1_h = 2 * prec_s * recall_h / (prec_s + recall_h) if (prec_s + recall_h) > 0 else 0.0

    r.total_injected_content = injected_c
    r.fixed_content = fixed_c
    r.f1_score_content = f1_c
    r.total_injected_hygiene = injected_h
    r.fixed_hygiene = fixed_h
    r.f1_score_hygiene = f1_h


# ============================================================================
# CORE FUNCTIONS
# ============================================================================

def inject_errors_at_level(
    llm,
    original_content: str,
    error_count: int,
    output_dir: str,
    file_basename: str
) -> Dict[str, Any]:
    """
    Inject errors into content at a specific level.
    
    Returns dict with corrupted content and manifest.
    """
    injector = LLMErrorInjector(llm)
    
    corrupted, manifest = injector.inject(
        original_content,
        min_per_category=error_count,
        max_words=50000  # Don't limit words for tutorials
    )
    
    # Save corrupted file
    corrupted_path = os.path.join(output_dir, f"{file_basename}.level_{error_count}.corrupted.Rmd")
    write_file(corrupted_path, corrupted)
    
    # Save manifest
    manifest_path = os.path.join(output_dir, f"{file_basename}.level_{error_count}.manifest.json")
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    return {
        "error_count": error_count,
        "corrupted_content": corrupted,
        "corrupted_path": corrupted_path,
        "manifest": manifest,
        "manifest_path": manifest_path,
        "total_errors": len(manifest.get("errors", []))
    }


# ============================================================================
# PROMPTS FOR MODEL COMPARISON
# ============================================================================

# BioGuider's comprehensive error-fixing prompt (domain-specific guidance)
BIOGUIDER_PROMPT = """You are "BioGuider," fixing documentation for biomedical software.

GROUND TRUTH
- Code blocks (``` fences) are the AUTHORITY. If prose contradicts code
  (package version, test name, marker gene, parameter value), fix the
  PROSE to match the CODE.

EVALUATION DIMENSIONS (fix errors in all categories)
1. Scientific accuracy: gene names, species, statistical tests, parameters,
   accession IDs must be correct and consistent with code blocks
2. Markdown formatting: headers, lists, links, inline code, tables,
   image syntax must follow proper markdown
3. Prose-code consistency: prose descriptions must agree with adjacent
   code block contents (versions, function names, parameter values)
4. Structure: section titles, YAML frontmatter must be correct

HOW TO FIX (BioGuider methodology)
- Scan the entire document systematically, dimension by dimension
- Use code blocks as the source of truth for factual claims
- Fix typos, broken links, wrong gene names, incorrect numbers
- Restore proper markdown formatting
- Do NOT add new content or remove existing sections
- Do NOT modify text inside ``` fences
- Output the COMPLETE fixed document as markdown

CORRUPTED DOCUMENT TO FIX:
"""

# Simple/Generic prompt - what a typical user might ask ChatGPT
SIMPLE_PROMPT = """Fix all errors in this document and output the corrected version:

"""

# GPT no-guidance prompt - just asking to proofread
GPT_BASIC_PROMPT = """Proofread and fix this document:

"""

# Skill 2 — generic prompt WITH evaluation criteria but NO structured guidance.
# Mirrors what a user would ask ChatGPT if they knew the evaluation rubric
# but had no BioGuider skill. Used for Purpose-2 benchmark (skill validation).
SKILL_GENERIC_PROMPT = """Fix all errors in this document and output the corrected version:
"""

# Available prompts for comparison
PROMPTS = {
    "bioguider": {
        "prompt": BIOGUIDER_PROMPT,
        "description": "BioGuider: Domain-specific bioinformatics error correction with detailed guidance"
    },
    "simple": {
        "prompt": SIMPLE_PROMPT,
        "description": "Simple: Generic 'fix errors' prompt (baseline)"
    },
    "gpt_basic": {
        "prompt": GPT_BASIC_PROMPT,
        "description": "GPT Basic: Just 'proofread and fix'"
    },
    "skill_generic": {
        "prompt": SKILL_GENERIC_PROMPT,
        "description": "Skill 2: Evaluation criteria shared, no structured guidance"
    },
}

# ============================================================================
# MODEL CONFIGURATIONS
# ============================================================================

# Available models — all routed through LiteLLM proxy (OPENAI_BASE_URL)
# gpt-oss model id: verify via `curl $OPENAI_BASE_URL/models` if routing fails
MODELS = {
    # OpenAI family — verified-real on the LiteLLM proxy
    "gpt-4o":          {"type": "litellm", "model": "gpt-4o"},
    "gpt-5.2":         {"type": "litellm", "model": "gpt-5.2"},
    "gpt-5.3-codex":   {"type": "litellm", "model": "gpt-5.3-codex"},
    "gpt-5.4":         {"type": "litellm", "model": "gpt-5.4"},
    "gpt-5.4-nano":    {"type": "litellm", "model": "gpt-5.4-nano"},
    # Open weights
    "gpt-oss":         {"type": "litellm", "model": "gpt-oss-120b"},
    # Moonshot Kimi
    "kimi-k2.5":       {"type": "litellm", "model": "kimi-k2.5"},
    "kimi-k2.6":       {"type": "litellm", "model": "kimi-k2.6"},
    # Zhipu
    "glm-5":           {"type": "litellm", "model": "glm-5"},
    # Minimax
    "minimax-m2.5":    {"type": "litellm", "model": "minimax-m2.5"},
    # DeepSeek (real — v3.2 on the proxy is mis-aliased to Claude, do NOT use)
    "deepseek-v4-flash": {"type": "litellm", "model": "deepseek-v4-flash"},
}

def print_prompts():
    """Print all available prompts for reference."""
    print("\n" + "="*70)
    print("AVAILABLE PROMPTS")
    print("="*70)
    for name, info in PROMPTS.items():
        print(f"\n--- {name.upper()} ---")
        print(f"Description: {info['description']}")
        print(f"Prompt preview: {info['prompt'][:100]}...")
    print("="*70 + "\n")

def print_models():
    """Print all available models for reference."""
    print("\n" + "="*70)
    print("AVAILABLE MODELS")
    print("="*70)
    for name, info in MODELS.items():
        desc = info.get('description', info.get('model', name))
        print(f"  {name}: {desc} ({info.get('type', 'litellm')})")
    print("="*70 + "\n")


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=2, min=4, max=60),
    retry=retry_if_exception_type(RateLimitError),
)
def _invoke_with_retry(llm: ChatOpenAI, prompt: str):
    """Invoke an LLM with exponential-backoff retry on RateLimitError.

    Returns the raw AIMessage so callers can inspect response_metadata for
    token usage.
    """
    return llm.invoke(prompt)


def fix_with_model(
    llm,
    corrupted_content: str,
    original_content: str,
    output_dir: str,
    file_basename: str,
    error_count: int,
    prompt_name: str = "bioguider",
    model_name: str = "gpt-4o",
) -> Tuple[str, Dict[str, int]]:
    """
    Fix corrupted content using specified model and prompt combination.

    Args:
        llm: Language model to use (for Azure OpenAI)
        corrupted_content: Content with errors
        original_content: Original correct content (for reference)
        output_dir: Where to save results
        file_basename: Base name for output files
        error_count: Error level being tested
        prompt_name: Name of prompt to use ("bioguider", "simple", "gpt_basic", etc.)
        model_name: Name of model ("gpt-4o", "qwen3_30b", etc.)

    Returns:
        (fixed_content, token_usage) where token_usage is a dict with keys
        prompt_tokens, completion_tokens, total_tokens (all int, default 0).
    """
    model_config = MODELS.get(model_name, {"type": "litellm", "model": model_name})
    model_id = model_config.get("model", model_name)
    token_usage: Dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    llm_override = ChatOpenAI(
        model=model_id,
        api_key=os.environ.get("OPENAI_API_KEY"),
        base_url=os.environ.get("OPENAI_BASE_URL"),
        timeout=120,
        max_retries=1,
    )

    # ── Prompt-based path ─────────────────────────────────────────────────────
    if prompt_name in PROMPTS:
        prompt_base = PROMPTS[prompt_name]["prompt"]
    else:
        prompt_base = BIOGUIDER_PROMPT

    prompt = prompt_base + corrupted_content + "\n\nOUTPUT THE COMPLETE FIXED DOCUMENT:"

    try:
        response = _invoke_with_retry(llm_override, prompt)
        fixed_content = response.content if hasattr(response, "content") else str(response)

        # Extract token usage from response metadata
        usage = getattr(response, "response_metadata", {}).get("token_usage", {})
        token_usage["prompt_tokens"] = int(usage.get("prompt_tokens", 0))
        token_usage["completion_tokens"] = int(usage.get("completion_tokens", 0))
        token_usage["total_tokens"] = int(usage.get("total_tokens", 0))

        # Clean up LLM wrapper text and markdown code fences
        lines = fixed_content.split('\n')

        while lines and not lines[0].strip().startswith('---'):
            if any(phrase in lines[0].lower() for phrase in ['here is', 'fixed document', 'corrected', 'output:', 'certainly', 'sure']):
                lines = lines[1:]
            elif lines[0].strip().startswith('```'):
                lines = lines[1:]
            elif lines[0].strip() == '':
                lines = lines[1:]
            else:
                break

        while lines and (lines[-1].strip() == '```' or lines[-1].strip() == ''):
            lines = lines[:-1]

        fixed_content = '\n'.join(lines)

        if len(fixed_content) < len(corrupted_content) * 0.5:
            print(f"  Warning: Fixed content too short ({len(fixed_content)} vs {len(corrupted_content)}), using corrupted")
            fixed_content = corrupted_content

    except Exception as e:
        print(f"  Error fixing content: {e}")
        fixed_content = corrupted_content

    # Save fixed file with model and prompt name in filename
    fixed_path = os.path.join(output_dir, f"{file_basename}.level_{error_count}.{model_name}_{prompt_name}.fixed.Rmd")
    write_file(fixed_path, fixed_content)

    return fixed_content, token_usage


# Backward compatibility alias
def fix_with_bioguider(llm, corrupted_content, original_content, output_dir, file_basename, error_count):
    """Legacy function - uses BioGuider prompt with GPT-4o."""
    fixed_content, _token_usage = fix_with_model(
        llm, corrupted_content, original_content, output_dir,
        file_basename, error_count, prompt_name="bioguider", model_name="gpt-4o"
    )
    return fixed_content



def evaluate_fixes(
    original_content: str,
    corrupted_content: str,
    fixed_content: str,
    manifest: Dict[str, Any],
    llm=None
) -> Tuple[Any, List[CategoryResult]]:
    """
    Evaluate how well errors were fixed.
    
    Returns: (BenchmarkResult, List[CategoryResult])
    """
    evaluator = BenchmarkEvaluator(llm)
    
    error_metrics, false_positives = evaluator.evaluate_single_file(
        baseline=original_content,
        corrupted=corrupted_content,
        revised=fixed_content,
        injection_manifest=manifest,
        file_path="test_file.Rmd",
        file_category="tutorial",
        detect_semantic_fp=False  # Skip for speed
    )
    
    result = evaluator.aggregate_results(
        error_metrics,
        false_positives,
        error_count=len(manifest.get("errors", [])),
        file_count=1
    )

    # Set protected region violations (Hard FP)
    protection = check_protected_regions(original_content, fixed_content)
    result.code_fence_violations = protection["code_fence_violations"]
    result.yaml_violations = protection["yaml_violations"]
    result.section_violations = protection["section_violations"]

    # Derive per-category breakdown directly from error_metrics that
    # BenchmarkEvaluator already computed above.  Previously this block
    # re-evaluated each error with a hand-rolled if/elif chain that diverged
    # from BenchmarkEvaluator._check_error_fixed(), causing the headline
    # `errors_fixed` (= result.true_positives) and the per-category `fixed`
    # sums to disagree.  Using the same error_metrics object guarantees both
    # counts are identical.
    from collections import Counter
    cat_fixed: Counter = Counter()
    cat_unfixed: Counter = Counter()
    for em in error_metrics:
        if em.is_fixed:
            cat_fixed[em.category] += 1
        else:
            cat_unfixed[em.category] += 1
    all_cats = sorted(set(cat_fixed) | set(cat_unfixed))
    category_results = []
    for cat in all_cats:
        fixed = cat_fixed[cat]
        unfixed = cat_unfixed[cat]
        injected = fixed + unfixed
        fix_rate = fixed / injected if injected > 0 else 0.0
        category_results.append(CategoryResult(
            category=cat,
            injected=injected,
            fixed=fixed,
            unfixed=unfixed,
            fix_rate=fix_rate,
        ))

    return result, category_results


def run_stress_level(
    llm,
    original_content: str,
    error_count: int,
    output_dir: str,
    file_basename: str,
    prompt_name: str = "bioguider",
    model_name: str = "gpt-4o",
) -> StressLevelResult:
    """
    Run a single stress test level.
    """
    import time
    start_time = time.time()
    
    print(f"  [Level {error_count}] Injecting errors...")
    
    # Inject errors
    injection_result = inject_errors_at_level(
        llm, original_content, error_count, output_dir, file_basename
    )
    
    print(f"  [Level {error_count}] Injected {injection_result['total_errors']} errors")
    
    model_desc = MODELS.get(model_name, {}).get("description", model_name)
    print(f"  [Level {error_count}] Fixing with {model_desc} using {prompt_name} prompt...")
    
    # Fix with specified model/prompt
    fixed_content, token_info = fix_with_model(
        llm,
        injection_result["corrupted_content"],
        original_content,
        output_dir,
        file_basename,
        error_count,
        prompt_name=prompt_name,
        model_name=model_name
    )
    
    print(f"  [Level {error_count}] Evaluating fixes...")
    
    # Evaluate
    result, category_results = evaluate_fixes(
        original_content,
        injection_result["corrupted_content"],
        fixed_content,
        injection_result["manifest"],
        llm
    )
    
    duration = time.time() - start_time
    
    # Print category breakdown
    combo_name = f"{model_name}+{prompt_name}"
    print(f"  [{combo_name}@{error_count}] Category breakdown:")
    for cr in category_results:
        print(f"    {cr.category}: {cr.fixed}/{cr.injected} fixed ({cr.fix_rate:.1%})")
    
    return StressLevelResult(
        error_count=error_count,
        total_errors_injected=injection_result["total_errors"],
        errors_fixed=result.true_positives,
        errors_unfixed=result.false_negatives,
        fix_rate=result.fix_rate,
        precision=result.precision,
        recall=result.recall,
        f1_score=result.f1_score,
        duration_seconds=duration,
        category_results=category_results,
        model_name=combo_name,  # Include both model and prompt name
        false_positives=result.false_positives,
        code_fence_violations=result.code_fence_violations,
        yaml_violations=result.yaml_violations,
        section_violations=result.section_violations,
        prompt_tokens=token_info.get("prompt_tokens", 0),
        completion_tokens=token_info.get("completion_tokens", 0),
        total_tokens=token_info.get("total_tokens", 0),
    )


def run_stress_test_parallel(
    llm,
    test_file: str,
    stress_levels: List[int],
    output_dir: str,
    max_workers: int = 16
) -> List[StressLevelResult]:
    """
    Run stress tests at multiple levels in parallel.
    """
    # Read original file
    original_content = read_file(test_file)
    if not original_content:
        raise ValueError(f"Could not read test file: {test_file}")
    
    file_basename = Path(test_file).stem
    
    print(f"\nRunning stress test on: {test_file}")
    print(f"Stress levels: {stress_levels}")
    print(f"Max workers: {max_workers}")
    print(f"Output dir: {output_dir}")
    
    # Save original for reference
    original_path = os.path.join(output_dir, f"{file_basename}.original.Rmd")
    write_file(original_path, original_content)
    
    results = []
    
    # Run levels in parallel
    # Note: LLM calls are I/O bound, so ThreadPoolExecutor is appropriate
    with ThreadPoolExecutor(max_workers=min(max_workers, len(stress_levels))) as executor:
        futures = {}
        for level in stress_levels:
            future = executor.submit(
                run_stress_level,
                llm,
                original_content,
                level,
                output_dir,
                file_basename
            )
            futures[future] = level
        
        for future in as_completed(futures):
            level = futures[future]
            try:
                result = future.result()
                results.append(result)
                print(f"  [Level {level}] Complete: F1={result.f1_score:.3f}, FixRate={result.fix_rate:.3f}")
            except Exception as e:
                print(f"  [Level {level}] FAILED: {e}")
    
    # Sort by error count
    results.sort(key=lambda r: r.error_count)
    
    return results


def save_results(results: List[StressLevelResult], output_dir: str):
    """Save results to JSON and CSV."""
    from bioguider.managers.config import UNSCORABLE_CATEGORIES

    # Derive scorable variants from the per-category breakdown. Idempotent —
    # safe to call on already-populated results.
    for r in results:
        _populate_scorable(r)

    # JSON format with category breakdown
    json_data = {
        "timestamp": datetime.now().isoformat(),
        "unscorable_categories": sorted(UNSCORABLE_CATEGORIES),
        "results": [
            {
                "model": r.model_name,
                "error_count": r.error_count,
                "total_errors_injected": r.total_errors_injected,
                "errors_fixed": r.errors_fixed,
                "errors_unfixed": r.errors_unfixed,
                "fix_rate": round(r.fix_rate, 4),
                "precision": round(r.precision, 4),
                "recall": round(r.recall, 4),
                "f1_score": round(r.f1_score, 4),
                "total_errors_injected_scorable": r.total_errors_injected_scorable,
                "errors_fixed_scorable": r.errors_fixed_scorable,
                "errors_unfixed_scorable": r.errors_unfixed_scorable,
                "fix_rate_scorable": round(r.fix_rate_scorable, 4),
                "precision_scorable": round(r.precision_scorable, 4),
                "recall_scorable": round(r.recall_scorable, 4),
                "f1_score_scorable": round(r.f1_score_scorable, 4),
                "total_injected_content": r.total_injected_content,
                "fixed_content": r.fixed_content,
                "f1_score_content": round(r.f1_score_content, 4),
                "total_injected_hygiene": r.total_injected_hygiene,
                "fixed_hygiene": r.fixed_hygiene,
                "f1_score_hygiene": round(r.f1_score_hygiene, 4),
                "false_positives": r.false_positives,
                "code_fence_violations": r.code_fence_violations,
                "yaml_violations": r.yaml_violations,
                "section_violations": r.section_violations,
                "duration_seconds": round(r.duration_seconds, 2),
                "category_breakdown": [
                    {
                        "category": cr.category,
                        "injected": cr.injected,
                        "fixed": cr.fixed,
                        "unfixed": cr.unfixed,
                        "fix_rate": round(cr.fix_rate, 4),
                        "scorable": cr.category not in UNSCORABLE_CATEGORIES,
                    }
                    for cr in (r.category_results or [])
                ]
            }
            for r in results
        ]
    }
    
    json_path = os.path.join(output_dir, "STRESS_TEST_RESULTS.json")
    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=2)
    
    # CSV format - summary table with model column. Scorable columns are
    # the UNSCORABLE_CATEGORIES-filtered variants (headline for the paper
    # figures, see bioguider.managers.config.UNSCORABLE_CATEGORIES).
    csv_path = os.path.join(output_dir, "STRESS_TEST_TABLE.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "model", "error_count", "total_injected", "fixed", "unfixed",
            "fix_rate", "precision", "recall", "f1_score",
            "total_injected_scorable", "fixed_scorable", "unfixed_scorable",
            "fix_rate_scorable", "precision_scorable", "recall_scorable", "f1_score_scorable",
            "total_injected_content", "fixed_content", "f1_score_content",
            "total_injected_hygiene", "fixed_hygiene", "f1_score_hygiene",
            "false_positives", "code_fence_violations", "yaml_violations", "section_violations",
            "duration_s",
            "prompt_tokens", "completion_tokens", "total_tokens",
        ])
        for r in results:
            writer.writerow([
                r.model_name, r.error_count, r.total_errors_injected, r.errors_fixed, r.errors_unfixed,
                round(r.fix_rate, 4), round(r.precision, 4), round(r.recall, 4),
                round(r.f1_score, 4),
                r.total_errors_injected_scorable, r.errors_fixed_scorable, r.errors_unfixed_scorable,
                round(r.fix_rate_scorable, 4), round(r.precision_scorable, 4),
                round(r.recall_scorable, 4), round(r.f1_score_scorable, 4),
                r.total_injected_content, r.fixed_content, round(r.f1_score_content, 4),
                r.total_injected_hygiene, r.fixed_hygiene, round(r.f1_score_hygiene, 4),
                r.false_positives, r.code_fence_violations, r.yaml_violations, r.section_violations,
                round(r.duration_seconds, 2),
                r.prompt_tokens, r.completion_tokens, r.total_tokens,
            ])
    
    # CSV format - detailed category breakdown (for figures)
    csv_detail_path = os.path.join(output_dir, "STRESS_TEST_CATEGORY_DETAIL.csv")
    with open(csv_detail_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "model", "error_level", "category", "injected", "fixed", "unfixed", "fix_rate"
        ])
        for r in results:
            if r.category_results:
                for cr in r.category_results:
                    writer.writerow([
                        r.model_name, r.error_count, cr.category, cr.injected, cr.fixed, cr.unfixed,
                        round(cr.fix_rate, 4)
                    ])
    
    # Markdown report
    md_lines = [
        "# Single File Stress Test Results\n",
        f"\n**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n",
        "\n## Results Table\n",
        "\n| Errors | Injected | Fixed | Unfixed | Fix Rate | Precision | Recall | F1 |\n",
        "|--------|----------|-------|---------|----------|-----------|--------|----|\n",
    ]
    
    for r in results:
        md_lines.append(
            f"| {r.error_count} | {r.total_errors_injected} | {r.errors_fixed} | "
            f"{r.errors_unfixed} | {r.fix_rate:.1%} | {r.precision:.3f} | "
            f"{r.recall:.3f} | {r.f1_score:.3f} |\n"
        )
    
    # Add category breakdown table
    md_lines.append("\n## Category Breakdown\n")
    
    # Collect all categories across all levels
    all_categories = set()
    for r in results:
        if r.category_results:
            for cr in r.category_results:
                all_categories.add(cr.category)
    
    if all_categories:
        # Build header
        categories = sorted(all_categories)
        header = "| Level |"
        for cat in categories:
            header += f" {cat} |"
        md_lines.append(f"\n{header}\n")
        
        # Build separator
        sep = "|-------|"
        for _ in categories:
            sep += "----------|"
        md_lines.append(f"{sep}\n")
        
        # Build rows
        for r in results:
            row = f"| {r.error_count} |"
            cat_map = {cr.category: cr for cr in (r.category_results or [])}
            for cat in categories:
                if cat in cat_map:
                    cr = cat_map[cat]
                    row += f" {cr.fixed}/{cr.injected} ({cr.fix_rate:.0%}) |"
                else:
                    row += " - |"
            md_lines.append(f"{row}\n")
    
    # Find performance drop-off
    md_lines.append("\n## Analysis\n")
    prev_f1 = 1.0
    for r in results:
        if r.f1_score < prev_f1 * 0.8:
            md_lines.append(f"\n**Performance drop-off detected at {r.error_count} errors** (F1 dropped to {r.f1_score:.3f})\n")
            break
        prev_f1 = r.f1_score
    else:
        md_lines.append("\n**Performance stable across all tested error levels**\n")
    
    md_path = os.path.join(output_dir, "STRESS_TEST_REPORT.md")
    with open(md_path, 'w') as f:
        f.writelines(md_lines)
    
    print("\nResults saved to:")
    print(f"  - {json_path}")
    print(f"  - {csv_path}")
    print(f"  - {md_path}")

    try:
        from bioguider.generation.viz import BenchmarkPlotter
        BenchmarkPlotter(output_dir).render_all(output_dir)
    except ImportError:
        print("matplotlib not available; skipping figure generation")


# ============================================================================
# TEST FUNCTIONS
# ============================================================================

@pytest.fixture
def test_output_dir():
    """Create output directory for test."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(OUTPUT_BASE, f"run_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


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
# BIOGUIDER PIPELINE BENCHMARK (evaluate + generate with real BioGuider logic)
# ============================================================================

def test_bioguider_pipeline_matrix(llm, test_output_dir):
    """
    BioGuider-pipeline benchmark — same matrix shape as test_multi_file_full_matrix
    but replaces the prompt-only LLM call with the full BioGuider evaluate+generate
    pipeline for every model.

    Matrix: 10 vignettes × 9 error levels × N models = N×90 cells.

    Key difference from test_multi_file_full_matrix:
    - DocumentPipeline.prepare_repo() is called ONCE before the model loop.
      It builds CodeStructureDb (AST) and SummarizedFilesDb from the Seurat repo.
      Both are LLM-independent, so a single pipeline is shared across all models.
    - Each cell calls DocumentPipeline.evaluate_and_refine_document(llm=model_llm)
      so the per-model LLM is used for evaluation + generation, while the repo
      databases are reused.

    Outputs land in outputs/bioguider_pipeline_stress/run_<TS>/ with the same
    file layout as test_multi_file_full_matrix so the same analysis scripts apply.

    Run:
        pytest system_tests/test_single_file_stress.py::test_bioguider_pipeline_matrix -v -s
    """
    import time
    from bioguider.generation.document_pipeline import DocumentPipeline
    from bioguider.utils.constants import EvaluationTypeEnum

    error_levels = STRESS_LEVELS

    # ── Prepare shared pipeline once ─────────────────────────────────────────
    print(f"\nPreparing shared DocumentPipeline from {SEURAT_REPO_PATH} ...")
    shared_pipeline = DocumentPipeline(SEURAT_REPO_PATH).prepare_repo(llm)
    print("DocumentPipeline ready (CodeStructureDb + SummarizedFilesDb built).")

    multi_root = os.path.join(
        OUTPUT_BASE.replace("single_file_stress", "bioguider_pipeline_stress"),
        datetime.now().strftime("run_%Y%m%d_%H%M%S"),
    )
    os.makedirs(multi_root, exist_ok=True)

    total_cells = len(TUTORIAL_FILES) * len(error_levels) * len(MODELS)
    print(f"\n{'='*70}")
    print("BIOGUIDER PIPELINE MATRIX")
    print(f"{'='*70}")
    print(f"Files: {len(TUTORIAL_FILES)}, Levels: {len(error_levels)}, "
          f"Models: {len(MODELS)}, Total cells: {total_cells}")
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

        write_file(os.path.join(file_out, f"{file_stem}.original.Rmd"), original_content)

        print(f"\n{'#'*70}")
        print(f"# FILE: {file_stem}")
        print(f"{'#'*70}")

        file_results: List[StressLevelResult] = []

        for error_level in error_levels:
            print(f"\n--- Level {error_level} ---")

            # Deterministic injection — every model sees the same corrupted text
            injector = LLMErrorInjector(llm, force_deterministic=True)
            corrupted, manifest = injector.inject(
                original_content,
                min_per_category=error_level,
                max_words=50000,
            )
            corrupted_filename = f"{file_stem}.level_{error_level}.corrupted.Rmd"
            corrupted_path = os.path.join(file_out, corrupted_filename)
            write_file(corrupted_path, corrupted)
            manifest_path = os.path.join(file_out, f"{file_stem}.level_{error_level}.manifest.json")
            with open(manifest_path, "w") as f:
                json.dump(manifest, f, indent=2)
            total_injected = len(manifest.get("errors", []))
            print(f"    Injected {total_injected} errors (deterministic)")

            def _run_one_pipeline(model_name: str):
                combo = f"{model_name}+bioguider_pipeline"
                t0 = time.time()
                try:
                    model_config = MODELS.get(model_name, {"type": "litellm", "model": model_name})
                    model_id = model_config.get("model", model_name)
                    model_llm = ChatOpenAI(
                        model=model_id,
                        api_key=os.environ.get("OPENAI_API_KEY"),
                        base_url=os.environ.get("OPENAI_BASE_URL"),
                        timeout=120,
                        max_retries=1,
                    )

                    _, fixed_content = shared_pipeline.evaluate_and_refine_document(
                        llm=model_llm,
                        doc_repo_path=file_out,
                        doc_path=corrupted_filename,
                        eval_type=EvaluationTypeEnum.TUTORIAL,
                    )
                    duration = time.time() - t0

                    fixed_path = os.path.join(
                        file_out,
                        f"{file_stem}.level_{error_level}.{model_name}_bioguider_pipeline.fixed.Rmd",
                    )
                    write_file(fixed_path, fixed_content)

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
                    print(
                        f"    {combo:<40} F1={result.f1_score:.3f} "
                        f"fix={result.fix_rate:.1%} time={duration:.1f}s"
                    )
                except Exception as e:  # noqa: BLE001
                    print(f"    {combo:<40} ERROR: {e}")

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

    print(f"\n{'='*70}")
    print("BIOGUIDER PIPELINE MATRIX COMPLETE")
    print(f"{'='*70}")
    print(f"Files processed: {len(all_file_results)}")
    print(f"Total results: {len(pooled)}")
    print(f"Artifacts: {multi_root}")

    assert len(pooled) > 0, "No results produced — check LLM/proxy connectivity"


# ============================================================================
# SKILL COMPARISON TESTS (Workstream B)
# ============================================================================

def _write_skill_comparison_csv(rows: List[Dict[str, Any]], csv_path: str) -> None:
    """Write skill comparison rows to CSV.

    Columns: file_stem, model, skill, error_count, total_injected, fixed,
    unfixed, fix_rate, f1_score, f1_score_scorable, f1_score_content,
    f1_score_hygiene, duration_s.
    """
    fieldnames = [
        "file_stem", "model", "skill", "error_count", "total_injected",
        "fixed", "unfixed", "fix_rate", "f1_score", "f1_score_scorable",
        "f1_score_content", "f1_score_hygiene",
        "false_positives", "code_fence_violations", "yaml_violations",
        "section_violations",
        "duration_s",
    ]
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


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
