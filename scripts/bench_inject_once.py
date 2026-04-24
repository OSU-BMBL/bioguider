"""Phase 1 — inject once, reuse everywhere.

Runs ``GenerationTestManagerV2.run_quant_test`` in ``phase="eval_only"``
with ``force_deterministic=True`` so every later model run in the
horizontal matrix (Phase 2) sees the same corrupted ground truth.

Example:

    poetry run python -m scripts.bench_inject_once \\
      --repo data/.adalflow/repos/seurat_Seurat \\
      --report outputs/seurat_Seurat/evaluation.json \\
      --total-errors 300 \\
      --out outputs/eval_only/seurat_300

Emits ``outputs/eval_only/seurat_300/EVALUATION_STATE.json`` +
``BENCHMARK_MANIFEST.json`` + ``*.original`` sidecars alongside the
corrupted files. Hand the state JSON path to ``bench_run_horizontal.py``.
"""
from __future__ import annotations

import argparse
import os
import sys

from scripts._bench_common import (
    DEFAULT_MODEL,
    load_env,
    make_llm,
    make_step_callback,
)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--repo", required=True, help="Path to the baseline repo (clone dir).")
    p.add_argument("--report", required=True, help="Path to the evaluation report JSON.")
    p.add_argument(
        "--total-errors",
        type=int,
        default=300,
        help="Target total scorable errors to inject (default: 300).",
    )
    p.add_argument("--out", required=True, help="Output directory for the eval_only artifact.")
    p.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=(
            "Model to instantiate for the manager (injection skips the LLM via "
            "force_deterministic, so this only affects any LLM-dependent helpers)."
        ),
    )
    args = p.parse_args()

    load_env()

    # Lazy-import so the script fails fast on env issues before dragging the
    # heavy manager module in.
    from bioguider.managers.config import min_per_category_from_total, SCORABLE_CATEGORIES
    from bioguider.managers.generation_test_manager_v2 import GenerationTestManagerV2

    llm = make_llm(args.model)
    cb = make_step_callback("inject_once")

    mgr = GenerationTestManagerV2(llm=llm, step_callback=cb)
    # Hard-wire deterministic injection so re-runs produce byte-identical
    # corrupted output across every model in Phase 2.
    mgr.injector.force_deterministic = True

    os.makedirs(args.out, exist_ok=True)
    # We pass target_total_errors via the injector-level translation by
    # precomputing min_per_category here — the test manager's signature
    # doesn't take target_total_errors directly.
    n_files = 1  # Conservative; refined per-file in the manager.
    min_per_cat = min_per_category_from_total(
        args.total_errors,
        n_files=max(1, n_files),
        n_categories=len(SCORABLE_CATEGORIES),
    )
    sys.stdout.write(
        f"[inject_once] target_total_errors={args.total_errors} -> min_per_category={min_per_cat}\n"
    )

    result = mgr.run_quant_test(
        report_path=args.report,
        baseline_repo_path=args.repo,
        tmp_repo_path=args.out,
        min_per_category=min_per_cat,
        phase="eval_only",
    )
    sys.stdout.write(f"[inject_once] eval_only artifact: {result}\n")

    state_path = os.path.join(result, "EVALUATION_STATE.json")
    if not os.path.exists(state_path):
        sys.stderr.write(f"ERROR: expected state file not found: {state_path}\n")
        return 1
    sys.stdout.write(f"[inject_once] DONE — resume_from={state_path}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
