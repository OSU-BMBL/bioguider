"""Offline rescorer for multi-file stress benchmark runs.

Reads on-disk *.manifest.json + *.fixed.Rmd files produced by a prior benchmark
run and re-evaluates fix quality using the current scorer logic (including the
inline_code count-comparison fix and the CONTENT/HYGIENE category split).

No LLM calls are made — purely CPU-bound.

Usage:
    poetry run python scripts/bench_rescore.py \\
        --run-dir outputs/multi_file_stress/run_20260424_025546
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from bioguider.generation.unified_metrics import _naked_count
from bioguider.managers.config import (
    CONTENT_CATEGORIES,
    HYGIENE_CATEGORIES,
    UNSCORABLE_CATEGORIES,
)

MODELS = ["gpt-5.4", "kimi-k2.5", "gpt-oss", "gpt-4o"]

RESCORE_HASH_FILE = ".rescore_hash"

RESCORED_AGG_NAME = "AGGREGATE_TABLE_RESCORED.csv"
RESCORED_STEM_NAME = "STRESS_TEST_TABLE_RESCORED.csv"

AGG_COLS = [
    "file_stem",
    "model",
    "error_count",
    "total_injected",
    "fixed",
    "unfixed",
    "fix_rate",
    "precision",
    "recall",
    "f1_score",
    "total_injected_scorable",
    "fixed_scorable",
    "f1_score_scorable",
    "total_injected_content",
    "fixed_content",
    "f1_score_content",
    "total_injected_hygiene",
    "fixed_hygiene",
    "f1_score_hygiene",
    "duration_s",
]


# ---------------------------------------------------------------------------
# Scorer — mirrors test_single_file_stress.py:evaluate_fixes() inline branch
# ---------------------------------------------------------------------------

def _is_fixed(cat: str, orig: str, mut: str, corrupted: str, fixed: str) -> bool:
    """Return True if the error described by (cat, orig, mut) was fixed.

    Logic mirrors test_single_file_stress.py lines 469-533 with:
    - inline_code uses count-comparison (task-1 fix, no rewrap substring check)
    - stat_test_misnaming / celltype_marker fall through to default (no LLM)
    """
    if cat in ("typo", "bio_term", "function"):
        if orig and orig in fixed:
            return True
        if mut and mut in fixed:
            return False
        return True  # neither found = rewritten

    if cat == "link":
        return bool(re.search(r"\[[^\]]+\]\([^\s)]+\)", fixed))

    if cat == "markdown_structure":
        def _md_issues(text: str) -> int:
            n = 0
            n += len(re.findall(r"^#{1,6}[^\s#]", text, re.M))
            n += len(re.findall(r"^[-*][^\s]", text, re.M))
            return n
        return _md_issues(fixed) < _md_issues(corrupted)

    if cat == "inline_code":
        raw = mut.strip("`") if mut else ""
        if not raw:
            return False
        return _naked_count(fixed, raw) < _naked_count(corrupted, raw)

    if cat == "duplicate":
        return fixed.count(mut) < corrupted.count(mut) if mut else False

    if cat in (
        "number", "boolean", "param_name", "comment_typo",
        "species_name", "gene_case",
        "reproducibility_drift", "analysis_hyperparam", "annotation_id_space",
        "accession_id_prefix",
    ):
        if orig and orig in fixed:
            return True
        if mut and mut in fixed:
            return False
        return True  # neither found = rewritten

    if cat in ("stat_test_misnaming", "celltype_marker"):
        if orig and orig in fixed:
            return True
        if mut and mut not in fixed:
            return True
        return False  # no LLM fallback in offline mode

    # default: mutated gone or original restored
    return bool((mut and mut not in fixed) or (orig and orig in fixed))


def _f1(tp: int, fp: int, fn: int) -> float:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    if (precision + recall) == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def score_cell(
    manifest: dict[str, Any],
    corrupted_text: str,
    fixed_text: str,
) -> dict[str, int]:
    """Score a single (file, level, model) cell.

    Returns counts bucketed into total / scorable / content / hygiene.
    No FPs: offline rescorer sets fp=0 (precision=1.0 assumption consistent
    with the original benchmark which detected zero FPs across the full run).
    """
    errors = manifest.get("errors", [])

    total = len(errors)
    fixed_all = 0
    fixed_sc = 0
    fixed_ct = 0
    fixed_hy = 0
    sc = 0  # scorable injected
    ct = 0  # content injected
    hy = 0  # hygiene injected

    for e in errors:
        cat = e.get("category", "unknown")
        orig = e.get("original_snippet", "")
        mut = e.get("mutated_snippet", "")
        result = _is_fixed(cat, orig, mut, corrupted_text, fixed_text)

        if result:
            fixed_all += 1

        if cat not in UNSCORABLE_CATEGORIES:
            sc += 1
            if result:
                fixed_sc += 1

        if cat in CONTENT_CATEGORIES:
            ct += 1
            if result:
                fixed_ct += 1
        elif cat in HYGIENE_CATEGORIES:
            hy += 1
            if result:
                fixed_hy += 1

    return {
        "total": total,
        "fixed_all": fixed_all,
        "unfixed_all": total - fixed_all,
        "scorable": sc,
        "fixed_sc": fixed_sc,
        "content": ct,
        "fixed_ct": fixed_ct,
        "hygiene": hy,
        "fixed_hy": fixed_hy,
    }


# ---------------------------------------------------------------------------
# Duration lookup from original AGGREGATE_TABLE.csv
# ---------------------------------------------------------------------------

def _load_durations(run_dir: Path) -> dict[tuple[str, str, int], float]:
    """Return {(file_stem, model_key, error_count): duration_s}."""
    agg_csv = run_dir / "_aggregate" / "AGGREGATE_TABLE.csv"
    durations: dict[tuple[str, str, int], float] = {}
    if not agg_csv.exists():
        return durations
    with agg_csv.open() as fh:
        for row in csv.DictReader(fh):
            model_raw = row.get("model", "")
            ec = int(row.get("error_count", 0))
            ds = float(row.get("duration_s", 0))
            # model_raw is like "gpt-4o+bioguider"; file_stem not in agg CSV
            # The agg CSV is per (model, error_count) summed across stems —
            # we need per-stem. Fall back: store keyed by (model_raw, ec).
            durations[(model_raw, ec)] = ds
    return durations


def _load_stem_durations(stem_dir: Path) -> dict[tuple[str, int], float]:
    """Return {(model_key, error_count): duration_s} from per-stem STRESS_TEST_TABLE.csv."""
    tbl = stem_dir / "STRESS_TEST_TABLE.csv"
    out: dict[tuple[str, int], float] = {}
    if not tbl.exists():
        return out
    with tbl.open() as fh:
        for row in csv.DictReader(fh):
            model_raw = row.get("model", "")
            ec = int(row.get("error_count", 0))
            ds = float(row.get("duration_s", 0))
            out[(model_raw, ec)] = ds
    return out


# ---------------------------------------------------------------------------
# Hash computation for idempotency
# ---------------------------------------------------------------------------

def _compute_hash(run_dir: Path) -> str:
    """SHA-256 over sorted manifest mtimes + fixed-file mtimes + scorer module hash."""
    h = hashlib.sha256()

    # Scorer module hash
    scorer_path = Path(__file__)
    h.update(scorer_path.read_bytes())

    # Manifest + fixed file mtimes (sorted for determinism)
    paths = sorted(run_dir.rglob("*.manifest.json")) + sorted(
        run_dir.rglob("*.fixed.Rmd")
    )
    for p in paths:
        h.update(str(p).encode())
        h.update(str(p.stat().st_mtime_ns).encode())

    return h.hexdigest()


# ---------------------------------------------------------------------------
# Per-stem writer
# ---------------------------------------------------------------------------

def _write_stem_csv(stem_dir: Path, rows: list[dict]) -> None:
    out = stem_dir / RESCORED_STEM_NAME
    with out.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=AGG_COLS)
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Main scoring loop
# ---------------------------------------------------------------------------

def _model_key_in_filename(model: str) -> str:
    """Convert MODELS list name to the token used in filenames."""
    return model.replace(".", ".").replace("-", "-")  # identity — kept for clarity


def _find_files(stem_dir: Path, stem: str, level: int, model: str) -> tuple[Path | None, Path | None, Path | None]:
    corrupted = stem_dir / f"{stem}.level_{level}.corrupted.Rmd"
    manifest = stem_dir / f"{stem}.level_{level}.manifest.json"
    fixed = stem_dir / f"{stem}.level_{level}.{model}_bioguider.fixed.Rmd"
    return (
        corrupted if corrupted.exists() else None,
        manifest if manifest.exists() else None,
        fixed if fixed.exists() else None,
    )


def _score_one(
    stem: str,
    stem_dir: Path,
    level: int,
    model: str,
    stem_durations: dict[tuple[str, int], float],
) -> dict | None:
    corrupted_p, manifest_p, fixed_p = _find_files(stem_dir, stem, level, model)
    if not (corrupted_p and manifest_p and fixed_p):
        return None

    corrupted_text = corrupted_p.read_text(encoding="utf-8", errors="replace")
    fixed_text = fixed_p.read_text(encoding="utf-8", errors="replace")
    with manifest_p.open() as fh:
        manifest = json.load(fh)

    counts = score_cell(manifest, corrupted_text, fixed_text)

    total = counts["total"]
    fixed_all = counts["fixed_all"]
    unfixed_all = counts["unfixed_all"]
    fix_rate = fixed_all / total if total > 0 else 0.0
    # precision=1.0: no FP detection in offline mode
    f1_all = _f1(fixed_all, 0, unfixed_all)

    sc = counts["scorable"]
    fixed_sc = counts["fixed_sc"]
    f1_sc = _f1(fixed_sc, 0, sc - fixed_sc)

    ct = counts["content"]
    fixed_ct = counts["fixed_ct"]
    f1_ct = _f1(fixed_ct, 0, ct - fixed_ct)

    hy = counts["hygiene"]
    fixed_hy = counts["fixed_hy"]
    f1_hy = _f1(fixed_hy, 0, hy - fixed_hy)

    model_csv_key = f"{model}+bioguider"
    duration = stem_durations.get((model_csv_key, level), 0.0)

    return {
        "file_stem": stem,
        "model": model_csv_key,
        "error_count": level,
        "total_injected": total,
        "fixed": fixed_all,
        "unfixed": unfixed_all,
        "fix_rate": round(fix_rate, 4),
        "precision": 1.0,
        "recall": round(fixed_all / total if total > 0 else 0.0, 4),
        "f1_score": round(f1_all, 4),
        "total_injected_scorable": sc,
        "fixed_scorable": fixed_sc,
        "f1_score_scorable": round(f1_sc, 4),
        "total_injected_content": ct,
        "fixed_content": fixed_ct,
        "f1_score_content": round(f1_ct, 4),
        "total_injected_hygiene": hy,
        "fixed_hygiene": fixed_hy,
        "f1_score_hygiene": round(f1_hy, 4),
        "duration_s": duration,
    }


def rescore(run_dir: Path) -> None:
    run_dir = run_dir.resolve()
    hash_file = run_dir / RESCORE_HASH_FILE
    current_hash = _compute_hash(run_dir)

    if hash_file.exists() and hash_file.read_text().strip() == current_hash:
        print("idempotent: no changes (hash matches). Skipping rescore.")
        return

    agg_dir = run_dir / "_aggregate"
    agg_dir.mkdir(exist_ok=True)

    # Collect all stem directories (non-aggregate subdirs)
    stem_dirs = sorted(
        d for d in run_dir.iterdir()
        if d.is_dir() and d.name != "_aggregate"
    )

    # Discover all (stem, level) pairs from manifest filenames
    cells: list[tuple[str, Path, int, str]] = []
    for stem_dir in stem_dirs:
        stem = stem_dir.name
        for manifest_p in sorted(stem_dir.glob(f"{stem}.level_*.manifest.json")):
            # Extract level from filename: <stem>.level_<N>.manifest.json
            name = manifest_p.name
            level_str = name.split(".level_")[1].split(".")[0]
            try:
                level = int(level_str)
            except ValueError:
                continue
            for model in MODELS:
                cells.append((stem, stem_dir, level, model))

    # Load per-stem duration tables
    stem_duration_cache: dict[str, dict[tuple[str, int], float]] = {}
    for stem_dir in stem_dirs:
        stem_duration_cache[stem_dir.name] = _load_stem_durations(stem_dir)

    all_rows: list[dict] = []
    stem_rows: dict[str, list[dict]] = defaultdict(list)

    print(f"Rescoring {len(cells)} cells with {min(8, len(cells))} workers...")

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {
            executor.submit(
                _score_one,
                stem,
                stem_dir,
                level,
                model,
                stem_duration_cache[stem],
            ): (stem, level, model)
            for stem, stem_dir, level, model in cells
        }
        for fut in as_completed(futures):
            stem, level, model = futures[fut]
            try:
                row = fut.result()
            except Exception as exc:
                print(f"  ERROR scoring {stem} level={level} model={model}: {exc}", file=sys.stderr)
                continue
            if row is not None:
                all_rows.append(row)
                stem_rows[stem].append(row)

    # Sort for deterministic output: by file_stem, model, error_count
    all_rows.sort(key=lambda r: (r["file_stem"], r["model"], r["error_count"]))
    for stem in stem_rows:
        stem_rows[stem].sort(key=lambda r: (r["model"], r["error_count"]))

    # Write per-stem CSVs
    for stem_dir in stem_dirs:
        stem = stem_dir.name
        if stem in stem_rows:
            _write_stem_csv(stem_dir, stem_rows[stem])
            print(f"  wrote {stem_dir.name}/{RESCORED_STEM_NAME} ({len(stem_rows[stem])} rows)")

    # Write aggregate CSV (all rows, but aggregate version omits file_stem from
    # the task spec output — spec says columns include file_stem so keep it)
    agg_out = agg_dir / RESCORED_AGG_NAME
    with agg_out.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=AGG_COLS)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"Wrote {agg_out} ({len(all_rows)} rows)")

    # Write idempotency hash
    hash_file.write_text(current_hash)
    print("Rescore complete.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        required=True,
        type=Path,
        help="Path to the benchmark run directory (contains _aggregate/ and per-vignette subdirs)",
    )
    args = parser.parse_args()
    rescore(args.run_dir)


if __name__ == "__main__":
    main()
