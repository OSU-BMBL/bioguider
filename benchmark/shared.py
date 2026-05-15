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
import logging

import pytest
from langchain_openai import ChatOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from openai import RateLimitError

from bioguider.generation.llm_injector import LLMErrorInjector
from bioguider.generation.benchmark_metrics import BenchmarkEvaluator, check_protected_regions
from bioguider.agents.agent_utils import read_file, write_file

logger = logging.getLogger(__name__)

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
    file_basename: str,
    file_type: str = "",
) -> Dict[str, Any]:
    """
    Inject errors into content at a specific level.

    ``file_type`` is the extension (".md", ".rmd", ".rst", ...). It is
    forwarded to ``LLMErrorInjector.inject`` so file-type-gated categories
    (code_func_name, cli_flag_typo, …) fire correctly; default ``""`` keeps
    legacy Rmd callers unchanged. The corrupted-file extension is also
    derived from ``file_type`` (falls back to ``.Rmd`` when unset to preserve
    the historical filename pattern used by Seurat callers).

    Returns dict with corrupted content and manifest.
    """
    injector = LLMErrorInjector(llm)

    corrupted, manifest = injector.inject(
        original_content,
        min_per_category=error_count,
        max_words=50000,  # Don't limit words for tutorials
        file_type=file_type,
    )

    out_ext = file_type.lstrip(".") if file_type else "Rmd"

    # Save corrupted file
    corrupted_path = os.path.join(output_dir, f"{file_basename}.level_{error_count}.corrupted.{out_ext}")
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

# Available models
# litellm models are routed through LiteLLM proxy (OPENAI_BASE_URL)
# anthropic models call the Anthropic API directly using CLAUDE_API_KEY
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
    # Anthropic — direct API via CLAUDE_API_KEY
    "claude-sonnet-4-6": {"type": "anthropic", "model": "claude-sonnet-4-6"},
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
    file_type: str = ".Rmd",
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
        file_type: Extension for the saved ``.fixed`` file (e.g. ``".Rmd"``,
            ``".md"``). Defaults to ``.Rmd`` so existing Seurat benchmarks
            keep their historical filenames. Markdown-oriented benchmarks
            (e.g. pharokka) should pass ``".md"``.

    Returns:
        (fixed_content, token_usage) where token_usage is a dict with keys
        prompt_tokens, completion_tokens, total_tokens (all int, default 0).
    """
    ext = file_type if file_type.startswith(".") else f".{file_type}"
    model_config = MODELS.get(model_name, {"type": "litellm", "model": model_name})
    model_id = model_config.get("model", model_name)
    model_type = model_config.get("type", "litellm")
    token_usage: Dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    if model_type == "anthropic":
        from langchain_anthropic import ChatAnthropic
        llm_override = ChatAnthropic(
            model=model_id,
            api_key=os.environ.get("CLAUDE_API_KEY"),
            timeout=300,
            max_retries=1,
            max_tokens=8192,
        )
    else:
        llm_override = ChatOpenAI(
            model=model_id,
            api_key=os.environ.get("OPENAI_API_KEY"),
            base_url=os.environ.get("OPENAI_BASE_URL"),
            timeout=300,
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

        # Extract token usage — OpenAI uses "token_usage", Anthropic uses "usage"
        meta = getattr(response, "response_metadata", {})
        usage = meta.get("token_usage") or meta.get("usage", {})
        prompt_tok   = int(usage.get("prompt_tokens") or usage.get("input_tokens", 0))
        complete_tok = int(usage.get("completion_tokens") or usage.get("output_tokens", 0))
        token_usage["prompt_tokens"]     = prompt_tok
        token_usage["completion_tokens"] = complete_tok
        token_usage["total_tokens"]      = prompt_tok + complete_tok

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
    fixed_path = os.path.join(
        output_dir,
        f"{file_basename}.level_{error_count}.{model_name}_{prompt_name}.fixed{ext}",
    )
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


def select_largest_rd_files(repo_path: str, n: int = 10) -> List[str]:
    """Return the N largest qualifying .Rd files from repo_path/man/.

    Scans the full man/ directory (no alphabetical cap), filters out stubs
    that lack \\description{} or \\arguments{}, ranks by byte size descending,
    and returns up to n paths.
    """
    man_dir = os.path.join(repo_path, "man")
    if not os.path.isdir(man_dir):
        return []
    candidates = []
    for fname in os.listdir(man_dir):
        if not fname.endswith(".Rd") or fname.startswith("."):
            continue
        fpath = os.path.join(man_dir, fname)
        try:
            content = open(fpath, encoding="utf-8", errors="replace").read()
        except OSError:
            continue
        if r"\description{" not in content or r"\arguments{" not in content:
            continue
        candidates.append((len(content), fpath))
    candidates.sort(reverse=True)
    return [fpath for _, fpath in candidates[:n]]
