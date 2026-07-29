"""Compute protection violations post-hoc from saved baseline/revised files.

Walks a multi_file_stress run dir and emits STRESS_TEST_PROTECTION.csv per file
plus a top-level _aggregate/AGGREGATE_PROTECTION.csv. Used to patch back data
when the original test_multi_file_full_matrix run forgot to propagate
code_fence_violations / yaml_violations / section_violations through
StressLevelResult into save_results.

Usage:
    poetry run python scripts/compute_post_hoc_protection.py [run_dir]

If run_dir is omitted, picks the latest outputs/multi_file_stress/run_*.
"""
from __future__ import annotations

import csv
import glob
import os
import re
import sys
from typing import Optional

from bioguider.generation.benchmark_metrics import check_protected_regions


FIXED_FILENAME_RE = re.compile(
    r"^(?P<stem>.+)\.level_(?P<level>\d+)\.(?P<model>[^_]+)_(?P<prompt>[^.]+)\.fixed\.Rmd$"
)


def _find_latest_run() -> Optional[str]:
    candidates = sorted(
        glob.glob("outputs/multi_file_stress/run_*"),
        key=os.path.getmtime,
    )
    return candidates[-1] if candidates else None


def process_file_dir(file_dir: str) -> list:
    """Compute protection violations for every fixed file in one file_dir."""
    file_stem = os.path.basename(file_dir.rstrip("/"))
    baseline_path = os.path.join(file_dir, f"{file_stem}.original.Rmd")
    if not os.path.exists(baseline_path):
        print(f"  SKIP {file_stem}: no .original.Rmd")
        return []

    with open(baseline_path, encoding="utf-8") as fh:
        baseline = fh.read()

    rows = []
    for fname in os.listdir(file_dir):
        m = FIXED_FILENAME_RE.match(fname)
        if not m:
            continue
        level = int(m.group("level"))
        model = m.group("model")
        prompt = m.group("prompt")
        with open(os.path.join(file_dir, fname), encoding="utf-8") as fh:
            revised = fh.read()
        v = check_protected_regions(baseline, revised)
        rows.append({
            "file_stem": file_stem,
            "model": f"{model}+{prompt}",
            "error_level": level,
            "code_fence_violations": v["code_fence_violations"],
            "yaml_violations": v["yaml_violations"],
            "section_violations": v["section_violations"],
        })
    return rows


def write_csv(rows: list, path: str) -> None:
    if not rows:
        return
    fieldnames = ["file_stem", "model", "error_level",
                  "code_fence_violations", "yaml_violations", "section_violations"]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    if len(sys.argv) > 1:
        run_dir = sys.argv[1]
    else:
        run_dir = _find_latest_run()
        if not run_dir:
            print("No multi_file_stress run found.", file=sys.stderr)
            return 1
    print(f"Processing: {run_dir}")

    all_rows = []
    for entry in sorted(os.listdir(run_dir)):
        sub = os.path.join(run_dir, entry)
        if not os.path.isdir(sub) or entry.startswith("_"):
            continue
        rows = process_file_dir(sub)
        if rows:
            per_file_csv = os.path.join(sub, "STRESS_TEST_PROTECTION.csv")
            write_csv(rows, per_file_csv)
            print(f"  {entry}: {len(rows)} rows -> {per_file_csv}")
            all_rows.extend(rows)

    if all_rows:
        agg_dir = os.path.join(run_dir, "_aggregate")
        os.makedirs(agg_dir, exist_ok=True)
        agg_csv = os.path.join(agg_dir, "AGGREGATE_PROTECTION.csv")
        write_csv(all_rows, agg_csv)
        print(f"\nAggregate: {len(all_rows)} rows -> {agg_csv}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
