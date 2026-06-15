"""
Plot the capability benchmark (tool-calling + structured-output) scorecards.

Reads ``capability_results.json`` from one or more run directories and renders a
grouped bar chart: one cluster of metrics per model/mode label. Clusters are
laid out in a fixed order — by LLM (LLM_ORDER), and within each LLM
``prompt`` before ``native`` (MODE_ORDER) — not by score.

Metrics shown:
  tool.selection, tool.args, tool.abstention, struct.schema_valid,
  struct.field_accuracy, struct.exact_match

Usage:
    # latest single run
    conda run -n bioguider python benchmark/plot_capabilities.py

    # merge several runs into one figure (later runs win on duplicate labels)
    conda run -n bioguider python benchmark/plot_capabilities.py \
        --runs outputs/capability_bench/run_A outputs/capability_bench/run_B \
        --out  outputs/capability_bench/capability_5llm.png
"""
import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Fixed cluster order. Models / modes not listed here are appended after.
LLM_ORDER = ["gpt-4o", "gpt-5.4", "kimi-k2.5", "glm-5", "gpt-oss"]
MODE_ORDER = ["prompt", "native"]

METRICS = [
    ("tool", "selection_rate", "tool.select"),
    ("tool", "args_rate", "tool.args"),
    ("tool", "abstention_rate", "tool.abstain"),
    ("struct", "schema_valid_rate", "struct.schema"),
    ("struct", "field_accuracy", "struct.field"),
    ("struct", "exact_match_rate", "struct.exact"),
]
COLORS = ["#4878D0", "#6FB1E0", "#9CCB8E", "#EE854A", "#F4A862", "#D65F5F"]


def _latest_run(base: str) -> str:
    runs = sorted(glob.glob(os.path.join(base, "run_*")))
    if not runs:
        raise SystemExit(f"no run_* directories under {base}")
    return runs[-1]


def _load_runs(run_dirs):
    """Load + merge cards from run dirs. On duplicate label, later run wins."""
    by_label = {}
    for rd in run_dirs:
        path = os.path.join(rd, "capability_results.json")
        for card in json.load(open(path)):
            by_label[card.get("label", card["model"])] = card
    return list(by_label.values())


def _order_key(card):
    m = card["model"]
    mode = card.get("mode", "native")
    li = LLM_ORDER.index(m) if m in LLM_ORDER else len(LLM_ORDER)
    mi = MODE_ORDER.index(mode) if mode in MODE_ORDER else len(MODE_ORDER)
    return (li, mi, m, mode)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="outputs/capability_bench",
                    help="directory containing run_* folders")
    ap.add_argument("--run", default=None,
                    help="single run dir (defaults to the latest)")
    ap.add_argument("--runs", nargs="+", default=None,
                    help="multiple run dirs to merge into one figure")
    ap.add_argument("--out", default=None, help="output PNG path")
    args = ap.parse_args()

    if args.runs:
        run_dirs = args.runs
    elif args.run:
        run_dirs = [args.run]
    else:
        run_dirs = [_latest_run(args.base)]

    cards = sorted(_load_runs(run_dirs), key=_order_key)
    labels = [f"{c['model']} ({c.get('mode', 'native')})" for c in cards]

    n_groups = len(labels)
    n_metrics = len(METRICS)
    x = np.arange(n_groups)
    width = 0.8 / n_metrics

    fig, ax = plt.subplots(figsize=(max(10, 1.5 * n_groups), 6))
    for i, (fam, key, label) in enumerate(METRICS):
        vals = [c[fam][key] for c in cards]
        ax.bar(x + i * width, vals, width, label=label, color=COLORS[i])

    ax.set_xticks(x + width * (n_metrics - 1) / 2)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("rate")
    ax.set_title("LLM capability benchmark — tool calling & structured output")
    ax.legend(ncol=6, fontsize=8, loc="lower center", bbox_to_anchor=(0.5, -0.32))
    ax.grid(axis="y", alpha=0.3)
    # light separators between LLM clusters
    for j in range(1, n_groups):
        if cards[j]["model"] != cards[j - 1]["model"]:
            ax.axvline(x[j] - 0.5 * width, color="0.85", lw=1, zorder=0)
    fig.tight_layout()

    out_path = args.out or os.path.join(run_dirs[-1], "capability_summary.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"wrote {out_path}  ({n_groups} clusters from {len(run_dirs)} run(s))")


if __name__ == "__main__":
    main()
