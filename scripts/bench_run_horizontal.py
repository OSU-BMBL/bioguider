"""Phase 2 — horizontal model comparison (Figure A).

Iterates the LiteLLM model roster, resuming the Phase-1 eval_only state via
``phase="correction_only"``. Each model writes its own
``GEN_TEST_RESULTS.json`` under ``--out/<model>/``; the final Panel A parser
in ``scripts/render_mock_figures.py`` reads these per-model JSONs.

Example:

    poetry run python -m scripts.bench_run_horizontal \\
      --resume-from outputs/eval_only/seurat_300/EVALUATION_STATE.json \\
      --out outputs/bench_horizontal_seurat_300 \\
      --models kimi-k2.5 glm-5 gpt-oss-120b gpt-5.4 gpt-4o

Add ``--include-claude-code`` later if Qin wants the Claude Code agent as a
row in the same matrix (separate script will drive that one, not this).
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import List

from scripts._bench_common import (
    LITELLM_MODELS,
    load_env,
    make_llm,
    make_step_callback,
)


def _run_single_model(
    model: str,
    resume_from: str,
    out_root: str,
) -> str:
    from bioguider.managers.generation_test_manager_v2 import GenerationTestManagerV2

    model_out = os.path.join(out_root, model.replace("/", "_"))
    os.makedirs(model_out, exist_ok=True)

    llm = make_llm(model)
    cb = make_step_callback(f"h:{model}")
    mgr = GenerationTestManagerV2(llm=llm, step_callback=cb)

    result_dir = mgr.run_quant_test(
        # report_path + baseline/tmp paths are read from the state file when
        # phase='correction_only'; the positional args are required but ignored.
        report_path="",
        baseline_repo_path="",
        tmp_repo_path=model_out,
        phase="correction_only",
        resume_from=resume_from,
    )
    return result_dir


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--resume-from", required=True, help="Path to EVALUATION_STATE.json from Phase 1.")
    p.add_argument("--out", required=True, help="Root output directory for the matrix.")
    p.add_argument(
        "--models",
        nargs="+",
        default=LITELLM_MODELS,
        help=f"Model names to sweep. Default: {LITELLM_MODELS}",
    )
    args = p.parse_args()

    if not os.path.exists(args.resume_from):
        sys.stderr.write(f"ERROR: resume-from does not exist: {args.resume_from}\n")
        return 2

    load_env()
    os.makedirs(args.out, exist_ok=True)

    successes: List[str] = []
    failures: List[str] = []
    for model in args.models:
        sys.stdout.write(f"\n========== horizontal: {model} ==========\n")
        try:
            result_dir = _run_single_model(model, args.resume_from, args.out)
            sys.stdout.write(f"[h:{model}] DONE -> {result_dir}\n")
            successes.append(model)
        except Exception as exc:  # noqa: BLE001 — capture & continue so one bad model doesn't kill the sweep
            sys.stderr.write(f"[h:{model}] FAILED: {exc}\n")
            failures.append(model)

    sys.stdout.write("\n========== horizontal summary ==========\n")
    sys.stdout.write(f"  ok:      {successes}\n")
    sys.stdout.write(f"  failed:  {failures}\n")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
