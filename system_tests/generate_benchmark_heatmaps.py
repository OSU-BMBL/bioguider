"""Generate F1 and fix-rate heatmaps for a pipeline_stress benchmark run.

The plots mirror the layout of the reference figures
``outputs/benchmark_f1_adjusted.png`` and ``outputs/benchmark_fix_rate_adjusted.png``:

  * F1 figure — side-by-side ``CONTENT F1`` and ``HYGIENE F1`` heatmaps.
    Rows are ``<model>+<strategy>`` combinations grouped by strategy
    (simple, bioguider, pipeline), columns are error counts.
  * Fix-rate figure — single heatmap with the same row layout and a
    coloured side-rail per strategy group.

By default, the categories ``code_func_name``, ``duplicate``, ``inline_code``,
and ``markdown_structure`` are excluded from the recomputed metrics — matching
the reference "adjusted" figures.  The plotted F1 and fix-rate numbers are
recomputed from the per-row ``category_breakdown``; the pre-aggregated
``f1_score_content`` / ``fix_rate_scorable`` columns are ignored because they
include the excluded categories.

Usage::

    python system_tests/generate_benchmark_heatmaps.py <run-dir> [--exclude cat1,cat2,...]

``<run-dir>`` must contain ``STRESS_TEST_RESULTS.json``.  Two PNGs are
written next to it:

  * ``benchmark_f1_heatmap.png``
  * ``benchmark_fix_rate_heatmap.png``
"""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from typing import Dict, FrozenSet, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


STRATEGY_ORDER = ("simple", "bioguider", "pipeline")

# Categories that the user asked us to drop from the "adjusted" heatmaps —
# noisy or pipeline-unrelated formatting hits that swamp the real signal.
DEFAULT_EXCLUDED_CATEGORIES: FrozenSet[str] = frozenset({
    "code_func_name",
    "duplicate",
    "inline_code",
    "markdown_structure",
})


def _content_hygiene_sets() -> Tuple[FrozenSet[str], FrozenSet[str]]:
    """Pull CONTENT/HYGIENE partitions from the manager config.

    Wrapped in a function so import failures (e.g. in a stripped-down
    sandbox) can be tolerated by the caller via a tight fallback.
    """
    from bioguider.managers.config import CONTENT_CATEGORIES, HYGIENE_CATEGORIES
    return frozenset(CONTENT_CATEGORIES), frozenset(HYGIENE_CATEGORIES)


def _split_model_strategy(model_str: str) -> Tuple[str, str]:
    """``"gpt-4o+pipeline"`` → ``("gpt-4o", "pipeline")``.  Falls back to
    ``("...", "")`` when the suffix is missing."""
    if "+" not in model_str:
        return model_str, ""
    base, strategy = model_str.rsplit("+", 1)
    return base, strategy.lower()


def _load_results(run_dir: str) -> List[dict]:
    path = os.path.join(run_dir, "STRESS_TEST_RESULTS.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"missing {path}")
    with open(path) as fh:
        d = json.load(fh)
    return d.get("results", [])


def _recompute_row_metrics(
    row: dict,
    excluded: FrozenSet[str],
    content_cats: FrozenSet[str],
    hygiene_cats: FrozenSet[str],
) -> Dict[str, float]:
    """Recompute fix-rate and content/hygiene F1 for one result row.

    Uses ``category_breakdown`` so excluded categories are dropped from
    both numerator (``fixed``) and denominator (``fixed + unfixed``).
    F1 uses the row's overall ``precision_scorable`` as the precision
    baseline (group-agnostic), matching how ``benchmark/shared.py``
    computes content/hygiene F1 for the pre-aggregated columns.

    Returns ``{"fix_rate": float, "f1_content": float, "f1_hygiene": float}``;
    ``nan`` for any metric whose denominator is empty after exclusion.
    """
    breakdown = row.get("category_breakdown") or []
    kept = [c for c in breakdown if c.get("category") not in excluded]

    fixed_total = sum(c.get("fixed", 0) for c in kept)
    unfixed_total = sum(c.get("unfixed", 0) for c in kept)
    injected_total = fixed_total + unfixed_total
    fix_rate = fixed_total / injected_total if injected_total > 0 else float("nan")

    fixed_c = sum(c.get("fixed", 0) for c in kept if c.get("category") in content_cats)
    unfixed_c = sum(c.get("unfixed", 0) for c in kept if c.get("category") in content_cats)
    injected_c = fixed_c + unfixed_c
    fixed_h = sum(c.get("fixed", 0) for c in kept if c.get("category") in hygiene_cats)
    unfixed_h = sum(c.get("unfixed", 0) for c in kept if c.get("category") in hygiene_cats)
    injected_h = fixed_h + unfixed_h

    recall_c = fixed_c / injected_c if injected_c > 0 else 0.0
    recall_h = fixed_h / injected_h if injected_h > 0 else 0.0
    prec = float(row.get("precision_scorable", row.get("precision", 1.0)) or 0.0)
    f1_c = (
        2 * prec * recall_c / (prec + recall_c)
        if (prec + recall_c) > 0 else 0.0
    )
    f1_h = (
        2 * prec * recall_h / (prec + recall_h)
        if (prec + recall_h) > 0 else 0.0
    )
    if injected_c == 0:
        f1_c = float("nan")
    if injected_h == 0:
        f1_h = float("nan")
    return {"fix_rate": fix_rate, "f1_content": f1_c, "f1_hygiene": f1_h}


def _grid_from_breakdown(
    results: List[dict],
    *,
    metric: str,
    excluded: FrozenSet[str],
    content_cats: FrozenSet[str],
    hygiene_cats: FrozenSet[str],
) -> Tuple[List[str], List[int], np.ndarray]:
    """Build a (rows, cols, matrix) triple from recomputed per-row metrics.

    ``metric`` ∈ {"fix_rate", "f1_content", "f1_hygiene"}.
    """
    by_cell: Dict[Tuple[str, int], float] = {}
    row_keys: set[str] = set()
    error_counts: set[int] = set()
    for r in results:
        model = r["model"]
        ec = int(r["error_count"])
        vals = _recompute_row_metrics(r, excluded, content_cats, hygiene_cats)
        val = vals[metric]
        by_cell[(model, ec)] = val
        row_keys.add(model)
        error_counts.add(ec)

    def _row_sort(k: str):
        base, strat = _split_model_strategy(k)
        try:
            idx = STRATEGY_ORDER.index(strat)
        except ValueError:
            idx = len(STRATEGY_ORDER)
        return (idx, base)

    rows = sorted(row_keys, key=_row_sort)
    cols = sorted(error_counts)
    matrix = np.full((len(rows), len(cols)), np.nan, dtype=float)
    for i, r_key in enumerate(rows):
        for j, ec in enumerate(cols):
            v = by_cell.get((r_key, ec))
            if v is not None:
                matrix[i, j] = v
    return rows, cols, matrix


def _strategy_group_runs(rows: List[str]) -> List[Tuple[str, int, int]]:
    """Return ``[(strategy_label, first_row_idx, last_row_idx), …]`` runs
    of consecutive rows that share a strategy suffix, used for the side
    annotation rail."""
    runs: List[Tuple[str, int, int]] = []
    current: str | None = None
    start = 0
    for i, k in enumerate(rows):
        _, strat = _split_model_strategy(k)
        if strat != current:
            if current is not None:
                runs.append((current, start, i - 1))
            current = strat
            start = i
    if current is not None:
        runs.append((current, start, len(rows) - 1))
    return runs


def _annotate_cells(ax, matrix: np.ndarray, fmt: str = "{:.2f}") -> None:
    rows_n, cols_n = matrix.shape
    for i in range(rows_n):
        for j in range(cols_n):
            v = matrix[i, j]
            if np.isnan(v):
                ax.text(j, i, "—", ha="center", va="center", fontsize=8, color="grey")
                continue
            color = "black" if 0.30 < v < 0.85 else "white"
            ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=8, color=color)


def _draw_heatmap(ax, rows, cols, matrix, *, title: str, cmap="RdYlGn"):
    im = ax.imshow(matrix, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(np.arange(len(cols)))
    ax.set_xticklabels(cols, fontsize=9)
    ax.set_yticks(np.arange(len(rows)))
    ax.set_yticklabels(rows, fontsize=8)
    ax.set_xlabel("Error count")
    ax.set_title(title, fontsize=11, fontweight="bold")
    _annotate_cells(ax, matrix)
    # Light horizontal separators between strategy groups.
    runs = _strategy_group_runs(rows)
    for _strat, _start, end in runs[:-1]:
        ax.axhline(end + 0.5, color="black", linewidth=1.0, alpha=0.6)
    return im


def generate_f1_heatmap(
    run_dir: str,
    results: List[dict],
    *,
    excluded: FrozenSet[str] = DEFAULT_EXCLUDED_CATEGORIES,
    content_cats: FrozenSet[str] | None = None,
    hygiene_cats: FrozenSet[str] | None = None,
) -> str:
    """Side-by-side CONTENT F1 / HYGIENE F1 heatmaps.

    F1 is recomputed per-row from ``category_breakdown`` after stripping
    ``excluded`` categories, so the plot matches the "adjusted" reference
    figure.
    """
    if content_cats is None or hygiene_cats is None:
        content_cats, hygiene_cats = _content_hygiene_sets()

    content_rows, content_cols, content = _grid_from_breakdown(
        results, metric="f1_content",
        excluded=excluded, content_cats=content_cats, hygiene_cats=hygiene_cats,
    )
    hyg_rows, hyg_cols, hygiene = _grid_from_breakdown(
        results, metric="f1_hygiene",
        excluded=excluded, content_cats=content_cats, hygiene_cats=hygiene_cats,
    )
    # Both grids already share row/col ordering via _grid_from_breakdown.
    rows, cols = content_rows, content_cols
    assert rows == hyg_rows and cols == hyg_cols, "grid axes drifted"

    fig_h = max(4.0, 0.45 * len(rows) + 2.0)
    fig, axes = plt.subplots(
        1, 2, figsize=(max(10, 2 + 1.3 * len(cols) * 2), fig_h),
        gridspec_kw={"wspace": 0.05},
    )
    _draw_heatmap(axes[0], rows, cols, content, title="A: CONTENT F1")
    im = _draw_heatmap(axes[1], rows, cols, hygiene, title="B: HYGIENE F1")
    axes[1].set_yticklabels([])  # rows are shared

    cbar = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.03)
    cbar.set_label("F1 (recomputed, adjusted)")

    fig.suptitle(
        "CONTENT vs HYGIENE F1 by Model and Error Level\n"
        f"(excl. {', '.join(sorted(excluded))})",
        fontsize=13, fontweight="bold", y=0.995,
    )
    out_path = os.path.join(run_dir, "benchmark_f1_heatmap.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def generate_fix_rate_heatmap(
    run_dir: str,
    results: List[dict],
    *,
    excluded: FrozenSet[str] = DEFAULT_EXCLUDED_CATEGORIES,
    content_cats: FrozenSet[str] | None = None,
    hygiene_cats: FrozenSet[str] | None = None,
) -> str:
    """Single heatmap with a strategy-rail on the right (simple/bioguider/pipeline).

    Fix rate is recomputed from ``category_breakdown`` after stripping
    ``excluded`` categories.
    """
    if content_cats is None or hygiene_cats is None:
        content_cats, hygiene_cats = _content_hygiene_sets()
    rows, cols, matrix = _grid_from_breakdown(
        results, metric="fix_rate",
        excluded=excluded, content_cats=content_cats, hygiene_cats=hygiene_cats,
    )

    fig_h = max(4.0, 0.45 * len(rows) + 1.5)
    fig, ax = plt.subplots(figsize=(max(6, 2 + 1.4 * len(cols)), fig_h))
    im = _draw_heatmap(ax, rows, cols,
                       np.where(np.isnan(matrix), 0.0, matrix),
                       title=(
                           "Fix Rate by Model, Strategy & Error Level\n"
                           f"(excl. {', '.join(sorted(excluded))})"
                       ))
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("Fix Rate (recomputed, adjusted)")

    # Strategy side-rail on the right.
    runs = _strategy_group_runs(rows)
    rail_x = len(cols) - 0.5 + 0.15  # just outside the plot area
    for strat, start, end in runs:
        ax.annotate(
            strat.upper(),
            xy=(rail_x, (start + end) / 2),
            xytext=(rail_x + 0.7, (start + end) / 2),
            xycoords="data",
            ha="left", va="center", fontsize=10, fontweight="bold",
            annotation_clip=False,
        )
    out_path = os.path.join(run_dir, "benchmark_fix_rate_heatmap.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _filter_to_level(results: List[dict], level: int) -> List[dict]:
    return [r for r in results if int(r.get("error_count", -1)) == level]


def generate_per_level_heatmaps(
    run_dir: str,
    results: List[dict],
    *,
    excluded: FrozenSet[str] = DEFAULT_EXCLUDED_CATEGORIES,
    content_cats: FrozenSet[str] | None = None,
    hygiene_cats: FrozenSet[str] | None = None,
) -> List[str]:
    """Emit one F1 + one fix-rate heatmap per error level, twice — once
    unadjusted (no category exclusion) and once adjusted (``excluded``
    stripped from numerator+denominator).

    Output filenames per level ``L``::

        benchmark_{L}_f1_heatmap.png
        benchmark_{L}_f1_heatmap_adjusted.png
        benchmark_{L}_fix_rate_heatmap.png
        benchmark_{L}_fix_rate_heatmap_adjusted.png

    Returns the list of written paths.  Safe to call on partial / empty
    results — levels with no rows are skipped silently.
    """
    if content_cats is None or hygiene_cats is None:
        content_cats, hygiene_cats = _content_hygiene_sets()

    levels = sorted({int(r["error_count"]) for r in results if "error_count" in r})
    written: List[str] = []

    variants = (
        ("", frozenset()),               # unadjusted — all categories
        ("_adjusted", excluded),          # adjusted — drop the four noisy ones
    )

    for level in levels:
        subset = _filter_to_level(results, level)
        if not subset:
            continue
        for suffix, drop_set in variants:
            f1_path = os.path.join(run_dir, f"benchmark_{level}_f1_heatmap{suffix}.png")
            fr_path = os.path.join(run_dir, f"benchmark_{level}_fix_rate_heatmap{suffix}.png")
            _write_f1_heatmap_for_level(
                subset, level, f1_path,
                excluded=drop_set, content_cats=content_cats, hygiene_cats=hygiene_cats,
                suffix_label=suffix,
            )
            _write_fix_rate_heatmap_for_level(
                subset, level, fr_path,
                excluded=drop_set, content_cats=content_cats, hygiene_cats=hygiene_cats,
                suffix_label=suffix,
            )
            written.extend([f1_path, fr_path])
    return written


def _write_f1_heatmap_for_level(
    results: List[dict],
    level: int,
    out_path: str,
    *,
    excluded: FrozenSet[str],
    content_cats: FrozenSet[str],
    hygiene_cats: FrozenSet[str],
    suffix_label: str,
) -> None:
    """Single-column F1 figure for one error level."""
    content_rows, _cc, content = _grid_from_breakdown(
        results, metric="f1_content",
        excluded=excluded, content_cats=content_cats, hygiene_cats=hygiene_cats,
    )
    hyg_rows, _hc, hygiene = _grid_from_breakdown(
        results, metric="f1_hygiene",
        excluded=excluded, content_cats=content_cats, hygiene_cats=hygiene_cats,
    )
    rows = content_rows
    assert rows == hyg_rows, "row order drifted between content and hygiene grids"
    cols = [level]

    fig_h = max(4.0, 0.45 * len(rows) + 2.0)
    fig, axes = plt.subplots(1, 2, figsize=(8, fig_h), gridspec_kw={"wspace": 0.05})
    _draw_heatmap(axes[0], rows, cols, content, title="A: CONTENT F1")
    im = _draw_heatmap(axes[1], rows, cols, hygiene, title="B: HYGIENE F1")
    axes[1].set_yticklabels([])
    cbar = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.03)
    cbar.set_label(f"F1 ({'adjusted' if excluded else 'all categories'})")

    title = f"CONTENT vs HYGIENE F1 — error level {level}"
    if excluded:
        title += f"\n(excl. {', '.join(sorted(excluded))})"
    fig.suptitle(title, fontsize=12, fontweight="bold", y=0.995)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _write_fix_rate_heatmap_for_level(
    results: List[dict],
    level: int,
    out_path: str,
    *,
    excluded: FrozenSet[str],
    content_cats: FrozenSet[str],
    hygiene_cats: FrozenSet[str],
    suffix_label: str,
) -> None:
    rows, _cols, matrix = _grid_from_breakdown(
        results, metric="fix_rate",
        excluded=excluded, content_cats=content_cats, hygiene_cats=hygiene_cats,
    )
    cols = [level]

    fig_h = max(4.0, 0.45 * len(rows) + 1.5)
    fig, ax = plt.subplots(figsize=(6, fig_h))
    title = f"Fix Rate — error level {level}"
    if excluded:
        title += f"\n(excl. {', '.join(sorted(excluded))})"
    im = _draw_heatmap(
        ax, rows, cols,
        np.where(np.isnan(matrix), 0.0, matrix),
        title=title,
    )
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label(f"Fix Rate ({'adjusted' if excluded else 'all categories'})")

    runs = _strategy_group_runs(rows)
    rail_x = len(cols) - 0.5 + 0.15
    for strat, start, end in runs:
        ax.annotate(
            strat.upper(),
            xy=(rail_x, (start + end) / 2),
            xytext=(rail_x + 0.7, (start + end) / 2),
            xycoords="data",
            ha="left", va="center", fontsize=10, fontweight="bold",
            annotation_clip=False,
        )
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _merge_replicate_rows(model: str, level: int, rows: List[dict]) -> dict:
    """Merge replicate result rows for one ``(model, level)`` into one row.

    Multiple benchmark runs at the same error level are replicates of the
    same ``<model>+<strategy>`` combos. To get a stable per-level signal we
    pool them: ``category_breakdown`` ``fixed``/``unfixed``/``injected`` are
    summed per category across replicates, and ``precision_scorable`` is
    averaged (it is the precision baseline the F1 recompute uses). The merged
    row carries only the fields ``_recompute_row_metrics`` reads.
    """
    cat_acc: Dict[str, Dict[str, int]] = defaultdict(
        lambda: {"fixed": 0, "unfixed": 0, "injected": 0}
    )
    precisions: List[float] = []
    for r in rows:
        for c in r.get("category_breakdown") or []:
            cat = c.get("category")
            if cat is None:
                continue
            acc = cat_acc[cat]
            acc["fixed"] += int(c.get("fixed", 0))
            acc["unfixed"] += int(c.get("unfixed", 0))
            acc["injected"] += int(c.get("injected", c.get("fixed", 0) + c.get("unfixed", 0)))
        precisions.append(float(r.get("precision_scorable", r.get("precision", 1.0)) or 0.0))

    breakdown = [
        {"category": cat, "fixed": v["fixed"], "unfixed": v["unfixed"], "injected": v["injected"]}
        for cat, v in sorted(cat_acc.items())
    ]
    mean_prec = float(np.mean(precisions)) if precisions else 1.0
    return {
        "model": model,
        "error_count": level,
        "category_breakdown": breakdown,
        "precision_scorable": mean_prec,
    }


def aggregate_runs_by_level(
    run_dirs: List[str], levels: FrozenSet[int] | None = None
) -> Dict[int, List[dict]]:
    """Group result rows from many run dirs by error level, merging replicates.

    Args:
        run_dirs: directories each containing ``STRESS_TEST_RESULTS.json``.
        levels:   if given, keep only these error levels.

    Returns ``{level: [merged_row_per_model, ...]}``. Run dirs missing the
    results file are skipped silently.
    """
    # level -> model -> [raw rows]
    pool: Dict[int, Dict[str, List[dict]]] = defaultdict(lambda: defaultdict(list))
    for run_dir in run_dirs:
        path = os.path.join(run_dir, "STRESS_TEST_RESULTS.json")
        if not os.path.exists(path):
            continue
        with open(path) as fh:
            results = json.load(fh).get("results", [])
        for r in results:
            ec = int(r.get("error_count", -1))
            if levels is not None and ec not in levels:
                continue
            pool[ec][r["model"]].append(r)

    merged: Dict[int, List[dict]] = {}
    for level, by_model in pool.items():
        merged[level] = [
            _merge_replicate_rows(model, level, rows)
            for model, rows in by_model.items()
        ]
    return merged


def generate_category_heatmaps_for_levels(
    base_dir: str,
    levels: List[int],
    *,
    out_dir: str | None = None,
    excluded: FrozenSet[str] = frozenset(),
    content_cats: FrozenSet[str] | None = None,
    hygiene_cats: FrozenSet[str] | None = None,
) -> List[str]:
    """Emit one CONTENT/HYGIENE F1 heatmap per requested error level.

    Scans ``base_dir/run_*`` for ``STRESS_TEST_RESULTS.json``, pools all
    replicate runs per level, and writes (per level) a two-panel figure
    ``category_f1_heatmap_level_<L>.png`` — panel A = CONTENT F1, panel B =
    HYGIENE F1, rows = ``<model>+<strategy>`` grouped by strategy.

    Args:
        base_dir: e.g. ``outputs/pipeline_stress``.
        levels:   error levels to render (e.g. ``[20, 40, 100, 150]``).
        out_dir:  where PNGs land (defaults to ``base_dir``).
        excluded: categories to drop from numerator+denominator (default: none).

    Returns the list of written PNG paths.
    """
    if content_cats is None or hygiene_cats is None:
        content_cats, hygiene_cats = _content_hygiene_sets()
    out_dir = out_dir or base_dir
    os.makedirs(out_dir, exist_ok=True)

    import glob

    run_dirs = sorted(glob.glob(os.path.join(base_dir, "run_*")))
    by_level = aggregate_runs_by_level(run_dirs, frozenset(levels))

    written: List[str] = []
    for level in levels:
        results = by_level.get(level)
        if not results:
            print(f"no runs found for error level {level} — skipping")
            continue
        out_path = os.path.join(out_dir, f"category_f1_heatmap_level_{level}.png")
        _write_f1_heatmap_for_level(
            results, level, out_path,
            excluded=excluded, content_cats=content_cats, hygiene_cats=hygiene_cats,
            suffix_label="",
        )
        written.append(out_path)
        print(f"wrote {out_path}  ({len(results)} model+strategy rows pooled)")
    return written


def _parse_argv(argv: List[str]) -> Tuple[str, FrozenSet[str]] | None:
    """Tiny arg parser: ``<run-dir> [--exclude cat1,cat2,...]``."""
    if len(argv) < 2:
        return None
    run_dir = argv[1]
    excluded = DEFAULT_EXCLUDED_CATEGORIES
    i = 2
    while i < len(argv):
        if argv[i] == "--exclude" and i + 1 < len(argv):
            excluded = frozenset(
                s.strip() for s in argv[i + 1].split(",") if s.strip()
            )
            i += 2
        else:
            return None
    return run_dir, excluded


def main(argv: List[str]) -> int:
    parsed = _parse_argv(argv)
    if parsed is None:
        print(__doc__)
        return 2
    run_dir, excluded = parsed
    results = _load_results(run_dir)
    if not results:
        print(f"no results in {run_dir}/STRESS_TEST_RESULTS.json")
        return 1
    print(f"excluded categories: {', '.join(sorted(excluded)) if excluded else '(none)'}")

    # Combined heatmaps (every error level in one figure each).
    combined = [
        generate_f1_heatmap(run_dir, results, excluded=excluded),
        generate_fix_rate_heatmap(run_dir, results, excluded=excluded),
    ]
    # Per-error-level heatmaps, both adjusted and unadjusted, four files per level.
    per_level = generate_per_level_heatmaps(run_dir, results, excluded=excluded)
    for path in combined + per_level:
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    # When run as a top-level script, ``bioguider`` lives one level up.
    _repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    if _repo_root not in sys.path:
        sys.path.insert(0, _repo_root)
    sys.exit(main(sys.argv))
