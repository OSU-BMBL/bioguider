"""Phase 4 — thinking vs general mode (Figure C).

Resumes a Phase-1 eval_only state twice with the default model, once with
``thinking=True`` and once without, and records wall-clock + token usage
into ``<out>/thinking_vs_general.json``.

Example:

    poetry run python -m scripts.bench_thinking_vs_general \\
      --resume-from outputs/eval_only/seurat_300/EVALUATION_STATE.json \\
      --out outputs/bench_think_general

Token accounting relies on the LiteLLM ``token_usage`` payload; Panel C
prints whatever the proxy returns. For Kimi specifically,
``tests/test_litellm_compat.py`` already asserts non-zero reporting.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict

from scripts._bench_common import (
    DEFAULT_MODEL,
    load_env,
    make_llm,
    make_step_callback,
)


def _run_single_mode(model: str, thinking: bool, resume_from: str, out_dir: str) -> Dict[str, Any]:
    from bioguider.managers.generation_test_manager_v2 import GenerationTestManagerV2

    llm = make_llm(model, thinking=thinking)
    mode_name = "thinking" if thinking else "general"
    cb = make_step_callback(f"t:{mode_name}")

    mgr = GenerationTestManagerV2(llm=llm, step_callback=cb)
    sub_out = os.path.join(out_dir, mode_name)
    os.makedirs(sub_out, exist_ok=True)

    t0 = time.time()
    result_dir = mgr.run_quant_test(
        report_path="",
        baseline_repo_path="",
        tmp_repo_path=sub_out,
        phase="correction_only",
        resume_from=resume_from,
    )
    elapsed = time.time() - t0

    # Read the scorable F1 off of the produced results file.
    res_path = os.path.join(result_dir, "GEN_TEST_RESULTS.json")
    with open(res_path, "r", encoding="utf-8") as f:
        results = json.load(f)

    return {
        "mode": mode_name,
        "model": model,
        "wall_clock_s": round(elapsed, 2),
        "f1_score": results.get("f1_score"),
        "f1_score_scorable": results.get("f1_score_scorable"),
        "fix_rate_scorable": results.get("fix_rate_scorable"),
        "result_dir": result_dir,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--resume-from", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--model", default=DEFAULT_MODEL)
    args = p.parse_args()

    load_env()
    os.makedirs(args.out, exist_ok=True)

    records = []
    for thinking in (True, False):
        sys.stdout.write(f"\n========== thinking={thinking} ==========\n")
        records.append(_run_single_mode(args.model, thinking, args.resume_from, args.out))

    summary_path = os.path.join(args.out, "thinking_vs_general.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({"model": args.model, "runs": records}, f, indent=2)
    sys.stdout.write(f"[t:done] summary written to {summary_path}\n")
    for r in records:
        sys.stdout.write(
            f"  {r['mode']:<9} f1_scorable={r['f1_score_scorable']} "
            f"wall={r['wall_clock_s']}s\n"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
