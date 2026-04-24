"""Phase 3 — F1-vs-error-count gradient (Figure B).

Drives ``BenchmarkManager.run_total_error_gradient`` for the default model
at the TOTAL_ERROR_LEVELS gradient (50 / 100 / 200 / 300). Each level does
its own inject + correct + score — the gradient is the point, so we can't
reuse a single injection here.

Example:

    poetry run python -m scripts.bench_run_gradient \\
      --repo data/.adalflow/repos/seurat_Seurat \\
      --report outputs/seurat_Seurat/evaluation.json \\
      --out outputs/bench_gradient_seurat

Optional: pass ``--model`` to gradient a non-default model (e.g. to produce
the Claude-Code competitor curve overlaid on the BioGuider+Kimi curve).
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
    p.add_argument("--repo", required=True)
    p.add_argument("--report", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument(
        "--levels",
        nargs="+",
        type=int,
        default=None,
        help="Override TOTAL_ERROR_LEVELS (default [50,100,200,300]).",
    )
    p.add_argument(
        "--deterministic",
        action="store_true",
        help="Force deterministic injection so the gradient is reproducible.",
    )
    args = p.parse_args()

    load_env()
    from bioguider.managers.benchmark_manager import BenchmarkManager
    from bioguider.managers.config import TOTAL_ERROR_LEVELS

    llm = make_llm(args.model)
    cb = make_step_callback(f"grad:{args.model}")

    mgr = BenchmarkManager(llm=llm, step_callback=cb)
    if args.deterministic:
        mgr.injector.force_deterministic = True

    levels = args.levels or TOTAL_ERROR_LEVELS
    sys.stdout.write(f"[grad] levels={levels}\n")

    os.makedirs(args.out, exist_ok=True)
    results = mgr.run_total_error_gradient(
        report_path=args.report,
        baseline_repo_path=args.repo,
        output_base_path=args.out,
        total_levels=levels,
    )
    for level, r in results.items():
        f1 = getattr(r.evaluation_result, "f1_score_scorable", r.evaluation_result.f1_score)
        sys.stdout.write(f"[grad] level={level} f1_scorable={f1:.3f} dir={r.output_dir}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
