"""
Aggregate all pharokka pipeline benchmark runs and produce F1 / fix-rate figures.

Two figure styles are generated:
  1. Original style  (pharokka_f1_fixrate_summary.png / _selected.png):
     X-axis = error level, bars grouped by strategy+model.

  2. Restructured style  (pharokka_strategy_llm_summary.png / _selected.png):
     2 rows (F1, fix-rate) × 5 columns (error level 20/40/100/150/200).
     Within each panel, X-axis has 3 strategy clusters (pipeline | bioguider | simple).
     Within each cluster: one bar per LLM.
     Line chart overlaid to connect the same LLM across strategy clusters.

Scans outputs/pipeline_stress/run_*/STRESS_TEST_RESULTS.json.
Model field format: "<llm>+<strategy>" e.g. "gpt-4o+pipeline".

Usage:
    conda run -n bioguider python benchmark/plot_pharokka_benchmark.py
    conda run -n bioguider python benchmark/plot_pharokka_benchmark.py --outdir outputs/pipeline_stress
"""
import argparse
import glob
import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import numpy as np

TARGET_LEVELS = [40, 100, 150, 200]
STRATEGY_ORDER = ["pipeline", "simple"]
STRATEGY_LABELS = {"pipeline": "BioGuider", "simple": "Prompt"}
STRATEGY_COLORS = {"pipeline": "#4878D0", "simple": "#EE854A"}
# glm-5 dropped: proxy deployment deprecated (HTTP 410), unusable for the benchmark.
LLM_ORDER = ["gpt-5.4", "gpt-4o", "kimi-k2.5", "gpt-oss"]

EXCLUDED_CATEGORIES = {"code_func_name", "duplicate", "markdown_structure"}

LLM_COLORS = {
    "gpt-oss":    "#1f77b4",
    "gpt-4o":     "#ff7f0e",
    "gpt-5.4":    "#2ca02c",
    "kimi-k2.5":  "#9467bd",
}
LLM_MARKERS = {
    "gpt-oss":    "o",
    "gpt-4o":     "s",
    "gpt-5.4":    "^",
    "kimi-k2.5":  "v",
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_all_results(pipeline_stress_dir: str):
    """Return (rows, paths) where rows is a flat list of result dicts."""
    pattern = os.path.join(pipeline_stress_dir, "run_*", "STRESS_TEST_RESULTS.json")
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No STRESS_TEST_RESULTS.json found under {pipeline_stress_dir}")
    rows = []
    for p in paths:
        with open(p) as fh:
            data = json.load(fh)
        for r in data.get("results", []):
            rows.append(r)
    return rows, paths


def _parse_model(model_str: str):
    """Split 'gpt-4o+pipeline' -> ('gpt-4o', 'pipeline')."""
    if "+" in model_str:
        llm, strategy = model_str.rsplit("+", 1)
        return llm, strategy
    return model_str, "unknown"


def _compute_selected_metrics(r: dict, excluded: set):
    cats = r.get("category_breakdown", [])
    included = [c for c in cats if c["category"] not in excluded]
    fixed = sum(c["fixed"] for c in included)
    injected = sum(c["injected"] for c in included)
    if injected == 0:
        return float("nan"), float("nan")
    fp = int(r.get("false_positives", 0))
    precision = fixed / (fixed + fp) if (fixed + fp) > 0 else 0.0
    recall = fixed / injected
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return f1, recall


# ---------------------------------------------------------------------------
# Aggregation (shared by both figure styles)
# ---------------------------------------------------------------------------

def aggregate_by_llm_strategy(rows, excluded: set = None):
    """
    Returns dict[(llm, strategy, level)] -> {"f1": [...], "fix": [...]}
    """
    excluded = excluded or set()
    data = defaultdict(lambda: {"f1": [], "fix": []})
    for r in rows:
        lvl = r["error_count"]
        if lvl not in TARGET_LEVELS:
            continue
        llm, strategy = _parse_model(r["model"])
        if strategy not in STRATEGY_ORDER:
            continue
        if excluded:
            f1, fix = _compute_selected_metrics(r, excluded)
        else:
            f1, fix = r["f1_score"], r["fix_rate"]
        if not (f1 != f1):
            data[(llm, strategy, lvl)]["f1"].append(f1)
            data[(llm, strategy, lvl)]["fix"].append(fix)
    return dict(data)


def aggregate_flat(rows, excluded: set = None):
    """
    Original aggregation: dict[(model_str, level)] -> list of values.
    Returns (strategies, levels, f1_map, fix_map).
    """
    excluded = excluded or set()
    f1_map = defaultdict(list)
    fix_map = defaultdict(list)
    for r in rows:
        lvl = r["error_count"]
        if lvl not in TARGET_LEVELS:
            continue
        strategy = r["model"]
        if excluded:
            f1, fix = _compute_selected_metrics(r, excluded)
        else:
            f1, fix = r["f1_score"], r["fix_rate"]
        if not (f1 != f1):
            f1_map[(strategy, lvl)].append(f1)
            fix_map[(strategy, lvl)].append(fix)
    strategies = sorted({k[0] for k in f1_map})
    levels = sorted({k[1] for k in f1_map})
    return strategies, levels, dict(f1_map), dict(fix_map)


# ---------------------------------------------------------------------------
# Original figure style
# ---------------------------------------------------------------------------

def _make_ax_original(ax, strategies, levels, data_map, ylabel, title, colors):
    x = np.arange(len(levels))
    width = 0.8 / max(len(strategies), 1)
    for i, strat in enumerate(strategies):
        means, errs = [], []
        for lvl in levels:
            vals = data_map.get((strat, lvl), [])
            means.append(np.mean(vals) if vals else float("nan"))
            errs.append(np.std(vals) if len(vals) > 1 else 0.0)
        offset = (i - len(strategies) / 2 + 0.5) * width
        ax.bar(x + offset, means, width * 0.9,
               yerr=errs, capsize=3,
               label=strat, color=colors[i % len(colors)], alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([str(l) for l in levels])
    ax.set_xlabel("Error level (errors per category)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)


def plot_original(strategies, levels, f1_map, fix_map, out_path: str, subtitle: str = ""):
    colors = plt.cm.tab20.colors  # type: ignore[attr-defined]
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    _make_ax_original(axes[0], strategies, levels, f1_map,
                      ylabel="F1 Score",
                      title="F1 Score by Error Level (mean ± std, N=5 runs)",
                      colors=colors)
    _make_ax_original(axes[1], strategies, levels, fix_map,
                      ylabel="Fix Rate",
                      title="Fix Rate by Error Level (mean ± std, N=5 runs)",
                      colors=colors)
    title = "Pharokka Pipeline Benchmark — Strategy Comparison"
    if subtitle:
        title += f"\n{subtitle}"
    fig.suptitle(title, fontsize=13, fontweight="bold")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, fontsize=7,
               bbox_to_anchor=(0.5, -0.02), framealpha=0.9)
    fig.tight_layout(rect=[0, 0.10, 1, 1])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved figure → {out_path}")


# ---------------------------------------------------------------------------
# Stacked figure style:
#   5 rows (error levels) stacked with NO gap, single metric per figure.
#   X-axis: 3 strategy clusters (Pipeline | Bioguider | Simple) with one
#   LLM point each; one line per cluster connecting the LLMs in order.
#   Cluster labels shown ONCE at the bottom (shared x-axis).
#   Error level numbers on the left margin of each row.
# ---------------------------------------------------------------------------

_CLUSTER_GAP = 1.5  # gap between clusters in LLM-index units


def _cluster_positions(n_llms):
    """Return (all_x_positions, cluster_centers, separator_xs) for 3 clusters."""
    step = n_llms + _CLUSTER_GAP
    all_xs, centers, seps = [], [], []
    for si in range(len(STRATEGY_ORDER)):
        offset = si * step
        xs = [offset + li for li in range(n_llms)]
        all_xs.extend(xs)
        centers.append(offset + (n_llms - 1) / 2)
        if si < len(STRATEGY_ORDER) - 1:
            seps.append(offset + n_llms - 1 + _CLUSTER_GAP / 2)
    return all_xs, centers, seps


def _draw_line_panel(ax, data, level, metric_key, ylim=None):
    """
    Draw one row's line chart into ax.  No cluster header text — drawn once
    externally.  Returns the list of present LLMs for caller use.
    """
    present_llms = [
        llm for llm in LLM_ORDER
        if any(data.get((llm, s, level), {}).get(metric_key) for s in STRATEGY_ORDER)
    ]
    n_llms = len(present_llms)
    if n_llms == 0:
        ax.set_visible(False)
        return []

    all_xs, centers, seps = _cluster_positions(n_llms)
    step = n_llms + _CLUSTER_GAP

    for si, strat in enumerate(STRATEGY_ORDER):
        offset = si * step
        strat_color = STRATEGY_COLORS[strat]

        xs, means, stds = [], [], []
        for li, llm in enumerate(present_llms):
            vals = data.get((llm, strat, level), {}).get(metric_key, [])
            mean = float(np.mean(vals)) if vals else float("nan")
            std = float(np.std(vals)) if len(vals) > 1 else 0.0
            xs.append(offset + li)
            means.append(mean)
            stds.append(std)

        xs_arr = np.array(xs)
        y_arr = np.array(means)
        s_arr = np.array(stds)
        valid = ~np.isnan(y_arr)

        # Line + error bars (±1 std); no markers
        if valid.sum() >= 2:
            ax.errorbar(
                xs_arr[valid], y_arr[valid], yerr=s_arr[valid],
                color=strat_color, linewidth=2.0, alpha=0.9,
                capsize=4, capthick=1.5, elinewidth=1.2,
                fmt="-", zorder=4,
            )

    # Vertical dashed separators between clusters
    for sx in seps:
        ax.axvline(sx, color="gray", linewidth=0.8, linestyle="--", alpha=0.45, zorder=1)

    ax.set_xticks(all_xs)
    ax.set_xticklabels([])           # bottom panel overrides this
    ax.set_xlim(-_CLUSTER_GAP / 2, all_xs[-1] + _CLUSTER_GAP / 2)
    if ylim is not None:
        ax.set_ylim(ylim[0], ylim[1])
        ax.set_yticks([0.8, 0.9, 1.0])
    ax.grid(axis="y", alpha=0.25)
    return present_llms


def _shared_legend(fig):
    handles = [
        Line2D([0], [0], color=STRATEGY_COLORS[s], linewidth=2.2, label=STRATEGY_LABELS[s])
        for s in STRATEGY_ORDER
    ]
    labels = [STRATEGY_LABELS[s] for s in STRATEGY_ORDER]
    fig.legend(handles, labels,
               loc="lower center", ncol=len(handles), fontsize=10,
               bbox_to_anchor=(0.5, 0.01), framealpha=0.9,
               title="Strategy", title_fontsize=9)


def plot_metric_stacked(data, metric_key, metric_title, out_path, subtitle="", ylim=None):
    """
    One metric, 5 error-level panels stacked with zero gap.
    Cluster labels (Pipeline | Bioguider | Simple) appear once at the bottom.
    Error level printed as a bold number on the left margin of each row.
    """
    n_rows = len(TARGET_LEVELS)
    panel_h = 2.6          # inches per row
    fig, axes = plt.subplots(
        n_rows, 1,
        figsize=(11, panel_h * n_rows),
        sharex=True,
        gridspec_kw={"hspace": 0},
    )

    present_llms = []
    for row, level in enumerate(TARGET_LEVELS):
        ax = axes[row]
        llms = _draw_line_panel(ax, data, level, metric_key, ylim=ylim)
        if llms:
            present_llms = llms

        # Error level label on left margin
        ax.text(-0.06, 0.5, str(level),
                transform=ax.transAxes, ha="right", va="center",
                fontsize=13, fontweight="bold", color="#222222")

        # Score y-label only on middle row
        if row == n_rows // 2:
            ax.set_ylabel("Score", fontsize=10, labelpad=6)

        # Suppress duplicate top border between panels
        if row > 0:
            ax.spines["top"].set_visible(False)

    # ---- Top: one shared cluster-name row above the first panel ----
    if present_llms:
        n_llms = len(present_llms)
        all_xs, centers, _ = _cluster_positions(n_llms)

        trans_top = axes[0].get_xaxis_transform()
        for strat, cx in zip(STRATEGY_ORDER, centers):
            axes[0].text(cx, 1.02, STRATEGY_LABELS[strat],
                         transform=trans_top, ha="center", va="bottom",
                         fontsize=13, fontweight="bold",
                         color=STRATEGY_COLORS[strat])

        # Thin underline below each cluster name
        for strat, cx in zip(STRATEGY_ORDER, centers):
            si = STRATEGY_ORDER.index(strat)
            x0 = si * (n_llms + _CLUSTER_GAP) - _CLUSTER_GAP / 2
            x1 = si * (n_llms + _CLUSTER_GAP) + n_llms - 1 + _CLUSTER_GAP / 2
            axes[0].annotate("", xy=(x1, 1.01), xytext=(x0, 1.01),
                             xycoords=trans_top, textcoords=trans_top,
                             arrowprops=dict(arrowstyle="-",
                                            color=STRATEGY_COLORS[strat],
                                            lw=1.5))

    # ---- Bottom panel: LLM tick labels only ----
    if present_llms:
        tick_labels = present_llms * len(STRATEGY_ORDER)
        axes[-1].set_xticklabels(tick_labels, rotation=38, ha="right", fontsize=8.5)

    # ---- Title & legend ----
    suptitle = f"Pharokka Pipeline Benchmark — {metric_title} (mean ± std, N=5 runs)"
    if subtitle:
        suptitle += f"\n{subtitle}"

    # Tighten top/bottom margins before placing title and legend
    fig.subplots_adjust(top=0.91, bottom=0.10, left=0.10, right=0.97)
    fig.suptitle(suptitle, fontsize=12, fontweight="bold")

    _shared_legend(fig)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved figure → {out_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--outdir",
        default="outputs/pipeline_stress",
        help="Directory containing run_* subdirs (default: outputs/pipeline_stress)",
    )
    parser.add_argument(
        "--style",
        choices=["original", "restructured", "both"],
        default="both",
        help="Which figure style(s) to generate",
    )
    args = parser.parse_args()

    rows, paths = load_all_results(args.outdir)
    print(f"Loaded {len(rows)} result rows from {len(paths)} run directories.")

    if args.style in ("original", "both"):
        strategies, levels, f1_map, fix_map = aggregate_flat(rows)
        print(f"Strategies: {strategies}")
        print(f"Error levels: {levels}")
        out_png = os.path.join(args.outdir, "pharokka_f1_fixrate_summary.png")
        plot_original(strategies, levels, f1_map, fix_map, out_png)

        print(f"\nGenerating selected-categories (original style)...")
        s_strats, s_levels, s_f1, s_fix = aggregate_flat(rows, excluded=EXCLUDED_CATEGORIES)
        out_sel = os.path.join(args.outdir, "pharokka_f1_fixrate_selected.png")
        excl_label = f"excluding: {', '.join(sorted(EXCLUDED_CATEGORIES))}"
        plot_original(s_strats, s_levels, s_f1, s_fix, out_sel, subtitle=excl_label)

    if args.style in ("restructured", "both"):
        data = aggregate_by_llm_strategy(rows)
        plot_metric_stacked(data, "f1", "F1 Score",
                            os.path.join(args.outdir, "pharokka_stacked_f1.png"))
        plot_metric_stacked(data, "fix", "Fix Rate",
                            os.path.join(args.outdir, "pharokka_stacked_fixrate.png"))

        print(f"\nGenerating selected-categories (stacked style)...")
        data_sel = aggregate_by_llm_strategy(rows, excluded=EXCLUDED_CATEGORIES)
        excl_label = f"excluding: {', '.join(sorted(EXCLUDED_CATEGORIES))}"
        _sel_ylim = (0.7, 1.02)
        plot_metric_stacked(data_sel, "f1", "F1 Score",
                            os.path.join(args.outdir, "pharokka_stacked_f1_selected.png"),
                            subtitle=excl_label, ylim=_sel_ylim)
        plot_metric_stacked(data_sel, "fix", "Fix Rate",
                            os.path.join(args.outdir, "pharokka_stacked_fixrate_selected.png"),
                            subtitle=excl_label, ylim=_sel_ylim)


if __name__ == "__main__":
    main()
