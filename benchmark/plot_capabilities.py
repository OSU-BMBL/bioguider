"""
Plot the capability benchmark (tool-calling + structured-output) scorecards.

Reads ``capability_results.json`` from a run directory and renders a grouped
bar chart: one cluster of metrics per model/mode label.

Metrics shown:
  tool.selection, tool.args, tool.abstention, struct.schema_valid,
  struct.field_accuracy, struct.exact_match

Usage:
    conda run -n bioguider python benchmark/plot_capabilities.py
    conda run -n bioguider python benchmark/plot_capabilities.py --run outputs/capability_bench/run_20260615_120000
"""
import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="outputs/capability_bench",
                    help="directory containing run_* folders")
    ap.add_argument("--run", default=None,
                    help="specific run dir (defaults to the latest)")
    args = ap.parse_args()

    run_dir = args.run or _latest_run(args.base)
    results_path = os.path.join(run_dir, "capability_results.json")
    with open(results_path) as f:
        cards = json.load(f)

    cards = sorted(cards, key=lambda c: c["overall"], reverse=True)
    labels = [c.get("label", c["model"]) for c in cards]

    n_groups = len(labels)
    n_metrics = len(METRICS)
    x = np.arange(n_groups)
    width = 0.8 / n_metrics

    fig, ax = plt.subplots(figsize=(max(8, 1.6 * n_groups), 5.5))
    for i, (fam, key, label) in enumerate(METRICS):
        vals = [c[fam][key] for c in cards]
        ax.bar(x + i * width, vals, width, label=label, color=COLORS[i])

    ax.set_xticks(x + width * (n_metrics - 1) / 2)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("rate")
    ax.set_title("LLM capability benchmark — tool calling & structured output")
    ax.legend(ncol=3, fontsize=8, loc="lower center", bbox_to_anchor=(0.5, -0.28))
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    out_path = os.path.join(run_dir, "capability_summary.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
