"""Phase 5 — correction-only sample of hand-picked files (qualitative figure).

Resumes a Phase-1 eval_only state with ``example_files=[…]`` so only those
files are corrected. Used for the paper's before/after diff figure (Qin's
"only two examples" decision on the call).

Example:

    poetry run python -m scripts.bench_pick_examples \\
      --resume-from outputs/eval_only/seurat_300/EVALUATION_STATE.json \\
      --out outputs/bench_examples \\
      --files vignettes/demo.Rmd README.md
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
    p.add_argument("--resume-from", required=True)
    p.add_argument("--out", required=True)
    p.add_argument(
        "--files",
        nargs="+",
        required=True,
        help="Relative paths (inside the injected repo) to correct.",
    )
    p.add_argument("--model", default=DEFAULT_MODEL)
    args = p.parse_args()

    load_env()
    from bioguider.managers.generation_test_manager_v2 import GenerationTestManagerV2

    llm = make_llm(args.model)
    cb = make_step_callback(f"pick:{args.model}")
    mgr = GenerationTestManagerV2(llm=llm, step_callback=cb)

    os.makedirs(args.out, exist_ok=True)
    result_dir = mgr.run_quant_test(
        report_path="",
        baseline_repo_path="",
        tmp_repo_path=args.out,
        phase="correction_only",
        resume_from=args.resume_from,
        example_files=args.files,
    )
    sys.stdout.write(f"[pick] corrected {len(args.files)} files -> {result_dir}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
