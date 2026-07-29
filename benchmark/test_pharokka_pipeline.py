"""
Benchmark: BioGuider pipeline vs. one-shot prompt on pharokka ``docs/plotting.md``.

Mirrors ``benchmark/test_stress.py::test_e004_pipeline_vs_prompt`` but targets
the pharokka user-guide page (Markdown). ``file_type=".md"`` is forwarded to
``inject_errors_at_level`` so the ``.md``-gated injection categories
(``cli_flag_typo``, ``cli_unknown_flag``, ``cli_program_rename``,
``code_func_name``, ``code_func_args``, ``code_comment_conflict``) actually
fire.

On first use the pharokka repo is cloned into
``data/.adalflow/repos/gbouras13_pharokka/`` via the same RAG helper our
system tests use. Subsequent runs reuse the existing clone.

Environment variables:
  PHAROKKA_MODELS       Comma-separated model keys
                        (default: gpt-4o,gpt-5.4,kimi-k2.5,glm-5.1,gpt-oss)
  PHAROKKA_ERROR_LEVEL  Errors per category to inject (default: 10)

Run:
    pytest benchmark/test_pharokka_pipeline.py::test_pipeline_vs_prompt_pharokka -v -s
    PHAROKKA_MODELS=gpt-4o PHAROKKA_ERROR_LEVEL=20 \\
        pytest benchmark/test_pharokka_pipeline.py::test_pipeline_vs_prompt_pharokka -v -s
"""
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed as _as_completed
from pathlib import Path
from typing import List

import pytest
from langchain_core.callbacks import UsageMetadataCallbackHandler
from langchain_openai import ChatOpenAI

from benchmark.shared import (
    MODELS,
    StressLevelResult,
    evaluate_fixes,
    fix_with_model,
    inject_errors_at_level,
    resolve_proxy_credentials,
    save_results,
)
from bioguider.agents.agent_utils import read_file, write_file
from bioguider.generation.document_pipeline import DocumentPipeline
from bioguider.utils.constants import EvaluationTypeEnum

PHAROKKA_REPO_URL = "https://github.com/gbouras13/pharokka"
PHAROKKA_REPO_DIR = "data/.adalflow/repos/gbouras13_pharokka"
PHAROKKA_DOC_REL = "docs/plotting.md"

# E004's default model list minus Claude (which is "claude-sonnet-4-6" in MODELS
# but already absent from the E004 default).
DEFAULT_MODELS = "gpt-4o,gpt-5.4,kimi-k2.5,glm-5.1,gpt-oss"


@pytest.fixture(scope="module")
def pharokka_repo_path() -> str:
    """Ensure pharokka is cloned to data/.adalflow/repos/gbouras13_pharokka/.

    Idempotent: if the clone already exists, the fixture reuses it. Skips
    the test if cloning fails or the target doc is missing afterwards.
    """
    if not os.path.isdir(PHAROKKA_REPO_DIR):
        try:
            from bioguider.rag.rag import RAG

            rag = RAG()
            rag.initialize_repo(repo_url_or_path=PHAROKKA_REPO_URL)
        except Exception as exc:  # pragma: no cover - network/IO
            pytest.skip(f"could not clone pharokka: {exc}")
    if not os.path.isdir(PHAROKKA_REPO_DIR):
        pytest.skip(f"pharokka clone missing at {PHAROKKA_REPO_DIR}")
    doc_path = os.path.join(PHAROKKA_REPO_DIR, PHAROKKA_DOC_REL)
    if not os.path.exists(doc_path):
        pytest.skip(f"{PHAROKKA_DOC_REL} not found in cloned pharokka repo")
    return PHAROKKA_REPO_DIR


def test_pipeline_vs_prompt_pharokka(llm, test_pipeline_output_dir, pharokka_repo_path):
    """Does the BioGuider pipeline beat one-shot prompts on pharokka's user guide?

    Injects errors once into ``docs/plotting.md``, then runs three strategies
    against the identical corrupted document for each model in
    ``PHAROKKA_MODELS``:
      - ``<model>+bioguider`` — direct LLM call with BioGuider's structured prompt
      - ``<model>+simple``    — direct LLM call with a one-line generic prompt
      - ``<model>+pipeline``  — ``DocumentPipeline`` evaluate→generate (eval_type=USERGUIDE)
    """
    test_output_dir = test_pipeline_output_dir
    target_models = [
        m.strip()
        for m in os.environ.get("PHAROKKA_MODELS", DEFAULT_MODELS).split(",")
        if m.strip()
    ]
    error_level = int(os.environ.get("PHAROKKA_ERROR_LEVEL", "10"))
    test_file = os.path.join(pharokka_repo_path, PHAROKKA_DOC_REL)

    original_content = read_file(test_file)
    file_basename = Path(test_file).stem  # "plotting"

    print(f"\n{'='*70}")
    print("PHAROKKA: PIPELINE vs BIOGUIDER PROMPT vs SIMPLE PROMPT")
    print(f"{'='*70}")
    print(f"File       : {test_file}")
    print(f"Models     : {target_models}")
    print(f"Error level: {error_level}")

    # ── Inject errors once — shared across all strategies ────────────────────
    injection_result = inject_errors_at_level(
        llm,
        original_content,
        error_level,
        test_output_dir,
        file_basename,
        file_type=".md",
    )
    total_injected = injection_result["total_errors"]
    corrupted_content = injection_result["corrupted_content"]
    corrupted_filename = f"{file_basename}.level_{error_level}.corrupted.md"
    manifest = injection_result["manifest"]
    print(f"Injected   : {total_injected} errors")
    write_file(
        os.path.join(test_output_dir, f"{file_basename}.original.md"),
        original_content,
    )

    # ── Build shared DocumentPipeline once (LLM-independent prep) ────────────
    print(f"\nPreparing DocumentPipeline from {pharokka_repo_path} ...")
    shared_pipeline = DocumentPipeline(pharokka_repo_path).prepare_repo(llm)
    print("DocumentPipeline ready.")

    all_results: List[StressLevelResult] = []

    # ── Strategy runners ─────────────────────────────────────────────────────

    def _run_prompt(model_name: str, prompt_name: str) -> "StressLevelResult | None":
        combo = f"{model_name}+{prompt_name}"
        t0 = time.time()
        try:
            fixed_content, token_usage = fix_with_model(
                llm,
                corrupted_content,
                original_content,
                test_output_dir,
                file_basename,
                error_level,
                prompt_name=prompt_name,
                model_name=model_name,
                file_type=".md",
            )
            duration = time.time() - t0
            result, category_results = evaluate_fixes(
                original_content, corrupted_content, fixed_content, manifest, llm
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
                prompt_tokens=token_usage.get("prompt_tokens", 0),
                completion_tokens=token_usage.get("completion_tokens", 0),
                total_tokens=token_usage.get("total_tokens", 0),
            )
            print(
                f"  {combo:<42} F1={result.f1_score:.3f} fix={result.fix_rate:.1%} "
                f"time={duration:.1f}s tokens={token_usage.get('total_tokens', 0)}"
            )
            return sr
        except Exception as e:
            print(f"  {combo:<42} ERROR: {e}")
            return None

    def _run_pipeline(model_name: str) -> "StressLevelResult | None":
        combo = f"{model_name}+pipeline"
        t0 = time.time()
        # Sum token usage across every internal LLM call the pipeline makes
        # (evaluation task + content generator + markdown polish). The handler
        # is attached at construction so it propagates to all .invoke calls on
        # this model instance, including with_structured_output / bind_tools.
        usage_cb = UsageMetadataCallbackHandler()
        # Per-call LLM timeout. Slow proxy models (glm-5.1, gpt-5.4) need far
        # more than the old 120s, especially at high error levels where the
        # generator emits a long document. Override with PHAROKKA_TIMEOUT.
        call_timeout = int(os.environ.get("PHAROKKA_TIMEOUT", "600"))
        # Retry transient proxy disconnects (APIConnectionError / "Server
        # disconnected"), which the tenacity layer does not catch (it only
        # retries 429s). Override with PHAROKKA_MAX_RETRIES.
        call_retries = int(os.environ.get("PHAROKKA_MAX_RETRIES", "4"))
        try:
            model_config = MODELS.get(model_name, {"type": "litellm", "model": model_name})
            model_id = model_config.get("model", model_name)
            model_type = model_config.get("type", "litellm")
            if model_type == "anthropic":
                from langchain_anthropic import ChatAnthropic
                model_llm = ChatAnthropic(
                    model=model_id,
                    api_key=os.environ.get("CLAUDE_API_KEY"),
                    timeout=call_timeout,
                    max_retries=call_retries,
                    max_tokens=8192,
                    callbacks=[usage_cb],
                )
            elif model_type == "azure":
                from langchain_openai import AzureChatOpenAI
                model_llm = AzureChatOpenAI(
                    azure_deployment=model_id,
                    azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT"),
                    api_key=os.environ.get("AZURE_OPENAI_API_KEY"),
                    api_version=os.environ.get("OPENAI_API_VERSION"),
                    timeout=call_timeout,
                    max_retries=call_retries,
                    callbacks=[usage_cb],
                )
            else:
                proxy_key, proxy_base_url = resolve_proxy_credentials()
                model_llm = ChatOpenAI(
                    model=model_id,
                    api_key=proxy_key,
                    base_url=proxy_base_url,
                    timeout=call_timeout,
                    max_retries=call_retries,
                    callbacks=[usage_cb],
                )
            report_path = os.path.join(
                test_output_dir,
                f"{file_basename}.level_{error_level}.{model_name}.pipeline_report.json",
            )
            eval_report_path = os.path.join(
                test_output_dir,
                f"{file_basename}.level_{error_level}.{model_name}.pipeline_eval.json",
            )
            _, fixed_content = shared_pipeline.evaluate_and_refine_document(
                llm=model_llm,
                doc_repo_path=test_output_dir,
                doc_path=corrupted_filename,
                eval_type=EvaluationTypeEnum.USERGUIDE,
                report_output_path=report_path,
                eval_report_output_path=eval_report_path,
            )
            duration = time.time() - t0
            write_file(
                os.path.join(
                    test_output_dir,
                    f"{file_basename}.level_{error_level}.{model_name}.pipeline_fixed.md",
                ),
                fixed_content,
            )
            result, category_results = evaluate_fixes(
                original_content, corrupted_content, fixed_content, manifest, llm
            )
            # Aggregate per-model token usage collected by the callback.
            prompt_tok = completion_tok = total_tok = 0
            for _m, u in (usage_cb.usage_metadata or {}).items():
                prompt_tok += int(u.get("input_tokens", 0))
                completion_tok += int(u.get("output_tokens", 0))
                total_tok += int(u.get("total_tokens", 0))
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
                prompt_tokens=prompt_tok,
                completion_tokens=completion_tok,
                total_tokens=total_tok,
            )
            print(
                f"  {combo:<42} F1={result.f1_score:.3f} fix={result.fix_rate:.1%} "
                f"time={duration:.1f}s tokens={total_tok}"
            )
            return sr
        except Exception as e:
            print(f"  {combo:<42} ERROR: {e}")
            return None

    # ── Run selected strategies × N models in parallel ───────────────────────
    # PHAROKKA_STRATEGIES selects which strategies run (comma-separated subset
    # of bioguider,simple,pipeline). Default runs all three. Use "pipeline" to
    # measure the full BioGuider pipeline's token usage + time in isolation.
    _all_strategies = ["bioguider", "simple", "pipeline"]
    strategies = [
        s.strip()
        for s in os.environ.get("PHAROKKA_STRATEGIES", ",".join(_all_strategies)).split(",")
        if s.strip() in _all_strategies
    ] or _all_strategies
    print(f"Strategies : {strategies}")
    # Cap concurrency: all models share one proxy endpoint, which throttles
    # (HTTP 429) and times out sub-calls when too many tasks run at once.
    # Override with PHAROKKA_MAX_WORKERS if needed.
    max_workers = int(os.environ.get("PHAROKKA_MAX_WORKERS", "4"))
    n_tasks = len(target_models) * len(strategies)
    print(f"\nRunning {n_tasks} tasks, {min(max_workers, n_tasks)} at a time ...")
    with ThreadPoolExecutor(max_workers=min(max_workers, n_tasks)) as pool:
        futures = {}
        for model_name in target_models:
            if "bioguider" in strategies:
                futures[pool.submit(_run_prompt, model_name, "bioguider")] = f"{model_name}+bioguider"
            if "simple" in strategies:
                futures[pool.submit(_run_prompt, model_name, "simple")] = f"{model_name}+simple"
            if "pipeline" in strategies:
                futures[pool.submit(_run_pipeline, model_name)] = f"{model_name}+pipeline"
        for future in _as_completed(futures):
            sr = future.result()
            if sr is not None:
                all_results.append(sr)

    save_results(all_results, test_output_dir)

    # ── Comparison table ─────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"PHAROKKA COMPARISON SUMMARY  (level={error_level}, file={file_basename})")
    print(f"{'='*70}")
    print(f"{'Strategy':<42} | {'F1':>6} | {'Fix%':>6} | {'Prec':>6} | {'Time':>7}")
    print("-" * 72)
    for sr in sorted(all_results, key=lambda r: -r.f1_score):
        print(
            f"{sr.model_name:<42} | {sr.f1_score:>6.3f} | {sr.fix_rate:>5.1%} "
            f"| {sr.precision:>6.3f} | {sr.duration_seconds:>6.1f}s"
        )

    # Per-model pipeline vs bioguider-prompt delta
    print("\n--- Pipeline vs BioGuider-prompt delta (per model) ---")
    for model_name in target_models:
        bio_r = next((r for r in all_results if r.model_name == f"{model_name}+bioguider"), None)
        pipe_r = next((r for r in all_results if r.model_name == f"{model_name}+pipeline"), None)
        if bio_r and pipe_r:
            delta = pipe_r.f1_score - bio_r.f1_score
            print(f"  {model_name}: pipeline {delta:+.3f} F1 vs bioguider prompt")

    assert all_results, "No results produced — check LLM/proxy connectivity"
