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
import re
import shutil
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple

import pytest

from bioguider.generation.llm_injector import LLMErrorInjector
from bioguider.generation.benchmark_metrics import BenchmarkEvaluator, BenchmarkResult
from bioguider.agents.agent_utils import read_file, write_file


# ============================================================================
# CONFIGURATION
# ============================================================================

# Default test file
DEFAULT_TEST_FILE = "data/.adalflow/repos/satijalab_seurat/vignettes/de_vignette.Rmd"

# Stress test levels (errors per category)
STRESS_LEVELS = [5, 10, 20, 40, 60, 100, 150, 200, 300]

# Quick test levels
QUICK_STRESS_LEVELS = [10, 40, 100]

# Output directory
OUTPUT_BASE = "outputs/single_file_stress"

# Max workers for parallel processing
MAX_WORKERS = 16


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
BIOGUIDER_PROMPT = """You are an expert document proofreader and error corrector for bioinformatics documentation.

Your task is to FIX ALL ERRORS in the following document. The document has been corrupted with:
- Typos and misspellings (e.g., "documnetation" should be "documentation")
- Truncated words (e.g., "Seura" should be "Seurat", "expre" should be "expression")
- Broken links (missing parts of URLs like "https//" should be "https://")
- Incorrect gene/protein names (wrong capitalization, misspellings)
- Incorrect species names (e.g., "humna" should be "human")
- Broken markdown formatting (headers, lists)
- Corrupted YAML frontmatter
- Changed boolean values (TRUE/FALSE swapped)
- Changed numeric values

CRITICAL RULES:
1. FIX every typo, truncated word, and misspelling you find
2. RESTORE broken URLs to their correct form
3. FIX gene names to proper capitalization (e.g., "brca1" -> "BRCA1")
4. PRESERVE all code blocks exactly - do not modify R code inside code fences
5. FIX the YAML header if it's corrupted (restore proper formatting)
6. Do NOT add new content - only fix existing errors
7. Do NOT remove any sections
8. Output the COMPLETE fixed document

CORRUPTED DOCUMENT TO FIX:
"""

# Simple/Generic prompt - what a typical user might ask ChatGPT
SIMPLE_PROMPT = """Fix all errors in this document and output the corrected version:

"""

# GPT no-guidance prompt - just asking to proofread
GPT_BASIC_PROMPT = """Proofread and fix this document:

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
    }
}

# ============================================================================
# MODEL CONFIGURATIONS
# ============================================================================

# API endpoints
OLLAMA_BASE_URL = "https://bmblx.bmi.osumc.edu/ollama"
CLAUDE_API_URL = "https://api.anthropic.com/v1/messages"

# Available models for comparison
MODELS = {
    # Azure OpenAI (current)
    "gpt4o": {
        "type": "azure",
        "description": "GPT-4o (Azure OpenAI)"
    },
    # Claude (Anthropic)
    "claude_sonnet": {
        "type": "claude",
        "model": "claude-sonnet-4-20250514",
        "description": "Claude Sonnet 4 (Anthropic)"
    },
    # Ollama models
    "qwen3_0.6b": {
        "type": "ollama",
        "model": "qwen3:0.6b",
        "description": "Qwen3 0.6B (fast, small)"
    },
    "qwen3_30b": {
        "type": "ollama",
        "model": "qwen3:30b",
        "description": "Qwen3 30B (balanced)"
    },
    "gpt_oss_20b": {
        "type": "ollama",
        "model": "gpt-oss:20b",
        "description": "GPT-OSS 20B (open source)"
    },
    "deepseek_r1_8b": {
        "type": "ollama",
        "model": "deepseek-r1:8b",
        "description": "DeepSeek-R1 8B (reasoning)"
    },
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
        print(f"  {name}: {info['description']} ({info['type']})")
    print("="*70 + "\n")


def call_ollama(model: str, prompt: str) -> str:
    """
    Call Ollama API to generate response.
    
    Args:
        model: Ollama model name (e.g., "qwen3:30b")
        prompt: The prompt to send
        
    Returns: Generated text response
    """
    import requests
    
    url = f"{OLLAMA_BASE_URL}/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "temperature": 0.3,  # Lower for more deterministic fixes
        "num_predict": 8000,  # Enough for full document
    }
    
    try:
        response = requests.post(url, json=payload, timeout=300)
        response.raise_for_status()
        result = response.json()
        return result.get("response", "")
    except Exception as e:
        print(f"  Ollama API error: {e}")
        return ""


def call_claude(model: str, prompt: str) -> str:
    """
    Call Claude API to generate response.
    
    Args:
        model: Claude model name (e.g., "claude-sonnet-4-20250514")
        prompt: The prompt to send
        
    Returns: Generated text response
    """
    import requests
    from dotenv import load_dotenv
    
    load_dotenv()
    api_key = os.environ.get("CLAUDE_API_KEY", "")
    
    if not api_key:
        print("  Warning: CLAUDE_API_KEY not found in environment")
        return ""
    
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json"
    }
    
    payload = {
        "model": model,
        "max_tokens": 8000,
        "messages": [
            {"role": "user", "content": prompt}
        ]
    }
    
    try:
        response = requests.post(CLAUDE_API_URL, headers=headers, json=payload, timeout=300)
        response.raise_for_status()
        result = response.json()
        # Claude returns content as a list of blocks
        content_blocks = result.get("content", [])
        if content_blocks and isinstance(content_blocks, list):
            return content_blocks[0].get("text", "")
        return ""
    except Exception as e:
        print(f"  Claude API error: {e}")
        return ""


def fix_with_model(
    llm,
    corrupted_content: str,
    original_content: str,
    output_dir: str,
    file_basename: str,
    error_count: int,
    prompt_name: str = "bioguider",
    model_name: str = "gpt4o",
) -> str:
    """
    Fix corrupted content using specified model and prompt combination.
    
    Args:
        llm: Language model to use (for Azure OpenAI)
        corrupted_content: Content with errors
        original_content: Original correct content (for reference)
        output_dir: Where to save results
        file_basename: Base name for output files
        error_count: Error level being tested
        prompt_name: Name of prompt to use ("bioguider", "simple", "gpt_basic")
        model_name: Name of model ("gpt4o", "qwen3_30b", etc.)
    
    Returns fixed content.
    """
    # Select prompt
    if prompt_name in PROMPTS:
        prompt_base = PROMPTS[prompt_name]["prompt"]
    else:
        prompt_base = BIOGUIDER_PROMPT
    
    # Build full prompt
    prompt = prompt_base + corrupted_content + "\n\nOUTPUT THE COMPLETE FIXED DOCUMENT:"
    
    # Get model config
    model_config = MODELS.get(model_name, {"type": "azure"})
    
    try:
        if model_config["type"] == "ollama":
            # Use Ollama API
            ollama_model = model_config["model"]
            fixed_content = call_ollama(ollama_model, prompt)
        elif model_config["type"] == "claude":
            # Use Claude API
            claude_model = model_config["model"]
            fixed_content = call_claude(claude_model, prompt)
        else:
            # Use Azure OpenAI (default)
            response = llm.invoke(prompt)
            fixed_content = response.content if hasattr(response, 'content') else str(response)
        
        # Clean up LLM wrapper text and markdown code fences
        lines = fixed_content.split('\n')
        
        # Remove common LLM intro lines
        while lines and not lines[0].strip().startswith('---'):
            if any(phrase in lines[0].lower() for phrase in ['here is', 'fixed document', 'corrected', 'output:', 'certainly', 'sure']):
                lines = lines[1:]
            elif lines[0].strip().startswith('```'):
                lines = lines[1:]
            elif lines[0].strip() == '':
                lines = lines[1:]
            else:
                break
        
        # Remove trailing code fence
        while lines and (lines[-1].strip() == '```' or lines[-1].strip() == ''):
            lines = lines[:-1]
        
        fixed_content = '\n'.join(lines)
        
        # Validate output length (should be similar to input)
        if len(fixed_content) < len(corrupted_content) * 0.5:
            print(f"  Warning: Fixed content too short ({len(fixed_content)} vs {len(corrupted_content)}), using corrupted")
            fixed_content = corrupted_content
            
    except Exception as e:
        print(f"  Error fixing content: {e}")
        fixed_content = corrupted_content
    
    # Save fixed file with model and prompt name in filename
    fixed_path = os.path.join(output_dir, f"{file_basename}.level_{error_count}.{model_name}_{prompt_name}.fixed.Rmd")
    write_file(fixed_path, fixed_content)
    
    return fixed_content


# Backward compatibility alias
def fix_with_bioguider(llm, corrupted_content, original_content, output_dir, file_basename, error_count):
    """Legacy function - uses BioGuider prompt with GPT-4o."""
    return fix_with_model(llm, corrupted_content, original_content, output_dir, 
                          file_basename, error_count, prompt_name="bioguider", model_name="gpt4o")


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
    
    # Calculate per-category results using SAME logic as BenchmarkEvaluator
    category_results = []
    errors = manifest.get("errors", [])
    
    # Group errors by category
    from collections import Counter
    category_counts = Counter(e.get("category", "unknown") for e in errors)
    
    # Count fixed vs unfixed per category
    category_fixed = Counter()
    category_unfixed = Counter()
    
    for e in errors:
        cat = e.get("category", "unknown")
        orig = e.get("original_snippet", "")
        mut = e.get("mutated_snippet", "")
        
        # Use same logic as BenchmarkEvaluator._check_error_fixed()
        is_fixed = False
        
        if cat in ("typo", "bio_term", "function"):
            # Fixed if: original in revised, OR neither in revised (rewritten)
            if orig and orig in fixed_content:
                is_fixed = True
            elif mut and mut in fixed_content:
                is_fixed = False
            else:
                is_fixed = True  # Neither found = rewritten = fixed
        
        elif cat == "link":
            # Fixed if any well-formed link exists
            is_fixed = bool(re.search(r"\[[^\]]+\]\([^\s)]+\)", fixed_content))
        
        elif cat == "markdown_structure":
            # Fixed if fewer markdown issues after
            def count_md_issues(text):
                issues = 0
                issues += len(re.findall(r"^#{1,6}[^\s#]", text, re.M))  # Header without space
                issues += len(re.findall(r"^[-*][^\s]", text, re.M))  # List without space
                return issues
            is_fixed = count_md_issues(fixed_content) < count_md_issues(corrupted_content)
        
        elif cat == "inline_code":
            # Fixed if rewrapped version present and mutated gone
            raw = mut.strip('`') if mut else ""
            rewrapped = f"`{raw}`" if raw else ""
            is_fixed = raw and rewrapped and rewrapped in fixed_content and mut not in fixed_content
        
        elif cat == "duplicate":
            # Fixed if fewer duplicates after
            is_fixed = fixed_content.count(mut) < corrupted_content.count(mut) if mut else False
        
        elif cat in ("number", "boolean", "param_name", "comment_typo", "species_name", "gene_case"):
            # For these: fixed if original restored OR mutated removed OR rewritten
            if orig and orig in fixed_content:
                is_fixed = True
            elif mut and mut in fixed_content:
                is_fixed = False
            else:
                is_fixed = True  # Neither found = rewritten
        
        else:
            # Default: mutated gone or original restored
            is_fixed = (mut and mut not in fixed_content) or (orig and orig in fixed_content)
        
        if is_fixed:
            category_fixed[cat] += 1
        else:
            category_unfixed[cat] += 1
    
    for cat in sorted(category_counts.keys()):
        injected = category_counts[cat]
        fixed = category_fixed[cat]
        unfixed = category_unfixed[cat]
        fix_rate = fixed / injected if injected > 0 else 0.0
        
        category_results.append(CategoryResult(
            category=cat,
            injected=injected,
            fixed=fixed,
            unfixed=unfixed,
            fix_rate=fix_rate
        ))
    
    return result, category_results


def run_stress_level(
    llm,
    original_content: str,
    error_count: int,
    output_dir: str,
    file_basename: str,
    prompt_name: str = "bioguider",
    model_name: str = "gpt4o",
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
    prompt_desc = PROMPTS.get(prompt_name, {}).get("description", prompt_name)
    print(f"  [Level {error_count}] Fixing with {model_desc} using {prompt_name} prompt...")
    
    # Fix with specified model/prompt
    fixed_content = fix_with_model(
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
        model_name=combo_name  # Include both model and prompt name
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
    
    # JSON format with category breakdown
    json_data = {
        "timestamp": datetime.now().isoformat(),
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
                "duration_seconds": round(r.duration_seconds, 2),
                "category_breakdown": [
                    {
                        "category": cr.category,
                        "injected": cr.injected,
                        "fixed": cr.fixed,
                        "unfixed": cr.unfixed,
                        "fix_rate": round(cr.fix_rate, 4)
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
    
    # CSV format - summary table with model column
    csv_path = os.path.join(output_dir, "STRESS_TEST_TABLE.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "model", "error_count", "total_injected", "fixed", "unfixed",
            "fix_rate", "precision", "recall", "f1_score", "duration_s"
        ])
        for r in results:
            writer.writerow([
                r.model_name, r.error_count, r.total_errors_injected, r.errors_fixed, r.errors_unfixed,
                round(r.fix_rate, 4), round(r.precision, 4), round(r.recall, 4),
                round(r.f1_score, 4), round(r.duration_seconds, 2)
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
    
    print(f"\nResults saved to:")
    print(f"  - {json_path}")
    print(f"  - {csv_path}")
    print(f"  - {md_path}")


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
    
    print(f"\nPrepared files for model comparison:")
    print(f"  - Corrupted file: {injection_result['corrupted_path']}")
    print(f"  - Errors: {injection_result['total_errors']}")
    print(f"  - Instructions: {instructions_path}")
    print(f"\nFix with each model and save to fixed_{{model}}/ directories")


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
        ("gpt4o", "bioguider"),      # GPT-4o with BioGuider prompt (should be best)
        ("gpt4o", "simple"),         # GPT-4o with simple prompt
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
            fixed_content = fix_with_model(
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
    print(f"\nInjecting errors...")
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
        
        fixed_content = fix_with_model(
            llm,
            injection_result["corrupted_content"],
            original_content,
            test_output_dir,
            file_basename,
            error_level,
            prompt_name=prompt_name,
            model_name="gpt4o"
        )
        
        # Evaluate
        result, category_results = evaluate_fixes(
            original_content,
            injection_result["corrupted_content"],
            fixed_content,
            injection_result["manifest"],
            llm
        )
        
        combo_name = f"gpt4o+{prompt_name}"
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
    """
    test_file = DEFAULT_TEST_FILE
    
    if not os.path.exists(test_file):
        pytest.skip(f"Test file not found: {test_file}")
    
    original_content = read_file(test_file)
    file_basename = Path(test_file).stem
    
    # Part 1: Model comparison at 30 errors only
    model_comparison_level = 30
    
    # Part 2: Stress test levels (BioGuider only)
    stress_levels = [10, 20, 40, 60, 100, 150, 200, 300]
    
    # Model configurations for comparison (at 30 errors)
    test_configs = [
        ("gpt4o", "bioguider"),      # BioGuider (should be best)
        ("gpt4o", "simple"),         # GPT-4o baseline
        ("claude_sonnet", "simple"), # Claude
        ("qwen3_30b", "simple"),     # Qwen 30B
        ("gpt_oss_20b", "simple"),   # GPT-OSS 20B
    ]


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
    
    # All model configurations
    test_configs = [
        ("gpt4o", "bioguider"),      # BioGuider
        ("gpt4o", "simple"),         # GPT-4o simple
        ("claude_sonnet", "simple"), # Claude
        ("qwen3_30b", "simple"),     # Qwen 30B
        ("gpt_oss_20b", "simple"),   # GPT-OSS 20B
    ]
    
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
                
                fixed_content = fix_with_model(
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
    
    print(f"\n--- AVERAGE PERFORMANCE BY MODEL ---")
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
    print(f"\n--- F1 SCORE BY MODEL AND ERROR LEVEL ---")
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
    
    print_prompts()
    print_models()
    
    print(f"\n{'='*70}")
    print("FULL BENCHMARK")
    print(f"{'='*70}")
    print(f"File: {test_file}")
    print(f"Part 1: Model comparison at {model_comparison_level} errors")
    print(f"Part 2: BioGuider stress test: {stress_levels}")
    
    # Save original
    write_file(os.path.join(test_output_dir, f"{file_basename}.original.Rmd"), original_content)
    
    all_results = []
    import time
    
    # =========================================================================
    # PART 1: Model comparison at 30 errors
    # =========================================================================
    print(f"\n{'='*70}")
    print(f"PART 1: MODEL COMPARISON @ {model_comparison_level} ERRORS")
    print(f"{'='*70}")
    
    # Inject errors for model comparison
    print(f"Injecting {model_comparison_level} errors per category...")
    injection_result = inject_errors_at_level(
        llm, original_content, model_comparison_level, test_output_dir, file_basename
    )
    print(f"Total injected: {injection_result['total_errors']} errors")
    
    # Test all models
    for model_name, prompt_name in test_configs:
        combo_name = f"{model_name}+{prompt_name}"
        print(f"\n--- {combo_name} ---")
        
        try:
            start_time = time.time()
            
            fixed_content = fix_with_model(
                llm,
                injection_result["corrupted_content"],
                original_content,
                test_output_dir,
                file_basename,
                model_comparison_level,
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
                error_count=model_comparison_level,
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
    
    # Save model comparison results
    save_results(all_results, test_output_dir)
    
    # =========================================================================
    # PART 2: BioGuider stress test (10-300 errors)
    # =========================================================================
    print(f"\n{'='*70}")
    print("PART 2: BIOGUIDER STRESS TEST (10-300 errors)")
    print(f"{'='*70}")
    
    for error_level in stress_levels:
        print(f"\n--- BioGuider @ Level {error_level} ---")
        
        # Inject errors for this level
        injection_result = inject_errors_at_level(
            llm, original_content, error_level, test_output_dir, file_basename
        )
        print(f"    Injected: {injection_result['total_errors']} errors")
        
        try:
            start_time = time.time()
            
            fixed_content = fix_with_model(
                llm,
                injection_result["corrupted_content"],
                original_content,
                test_output_dir,
                file_basename,
                error_level,
                prompt_name="bioguider",
                model_name="gpt4o"
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
                model_name="gpt4o+bioguider"
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
    print("FULL BENCHMARK SUMMARY")
    print(f"{'='*70}")
    
    # Part 1 results: Model comparison at 30 errors
    model_comp_results = [r for r in all_results if r.error_count == model_comparison_level]
    print(f"\n--- MODEL COMPARISON @ {model_comparison_level} errors ---")
    print(f"{'Model':<25} | {'Fixed':>6} | {'F1':>8} | {'FixRate':>8}")
    print("-" * 55)
    for r in sorted(model_comp_results, key=lambda x: x.f1_score, reverse=True):
        print(f"{r.model_name:<25} | {r.errors_fixed:>6} | {r.f1_score:>8.3f} | {r.fix_rate:>7.1%}")
    
    # Part 2 results: BioGuider stress test
    stress_results = [r for r in all_results if r.error_count != model_comparison_level]
    if stress_results:
        print(f"\n--- BIOGUIDER STRESS TEST ---")
        print(f"{'Errors':>8} | {'Fixed':>6} | {'F1':>8} | {'FixRate':>8}")
        print("-" * 40)
        for r in sorted(stress_results, key=lambda x: x.error_count):
            print(f"{r.error_count:>8} | {r.errors_fixed:>6} | {r.f1_score:>8.3f} | {r.fix_rate:>7.1%}")
    
    # Check if BioGuider is best in model comparison
    if model_comp_results:
        best = max(model_comp_results, key=lambda x: x.f1_score)
        bioguider_result = next((r for r in model_comp_results if "bioguider" in r.model_name), None)
        
        print(f"\n{'='*70}")
        if bioguider_result and bioguider_result == best:
            print(f"✓ BioGuider is BEST with F1={bioguider_result.f1_score:.3f}")
        elif bioguider_result:
            print(f"Best: {best.model_name} (F1={best.f1_score:.3f})")
            print(f"BioGuider: F1={bioguider_result.f1_score:.3f}")
    
    assert len(all_results) >= len(test_configs), "Should have model comparison results"
