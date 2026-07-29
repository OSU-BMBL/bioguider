"""
Plot token-usage and wall-time of the BioGuider pipeline across error levels.

Reads ``STRESS_TEST_TABLE.csv`` from every ``run_*`` under a pipeline_stress
directory (default ``outputs/pipeline_stress``), keeps only the ``+pipeline``
rows, and renders a two-panel figure:

    left  : total tokens vs error level   (one line per LLM)
    right : wall-time (s)  vs error level  (one line per LLM)

Multiple runs at the same (llm, level) are averaged. A companion
``pipeline_tokentime_summary.csv`` with the aggregated numbers is written next
to the figure.

Usage:
    conda run -n bioguider python benchmark/plot_pipeline_tokentime.py
    conda run -n bioguider python benchmark/plot_pipeline_tokentime.py \
        --base outputs/pipeline_stress --out outputs/pipeline_stress/pipeline_tokentime.png
"""
import argparse
import csv
import glob
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

TARGET_LEVELS = [40, 100, 150, 200]
# Fixed LLM order / styling, matching the other pharokka figures.
# Note: the Azure GPT-5.4 series is labeled "gpt-5.4-azure" in the CSVs, so its
# style key must match that exactly — otherwise it falls through to color=None
# and matplotlib auto-assigns a default-cycle color that collides with another
# line (previously it shared gpt-4o's color).
LLM_ORDER = ["gpt-5.4-azure", "gpt-5.4", "gpt-4o", "kimi-k2.5", "glm-5.1", "gpt-oss"]
LLM_COLORS = {
    "gpt-oss":       "#1f77b4",
    "gpt-4o":        "#ff7f0e",
    "gpt-5.4":       "#2ca02c",
    "gpt-5.4-azure": "#2ca02c",
    "kimi-k2.5":     "#9467bd",
    "glm-5.1":       "#d62728",
}
LLM_MARKERS = {
    "gpt-oss":       "o",
    "gpt-4o":        "s",
    "gpt-5.4":       "^",
    "gpt-5.4-azure": "^",
    "kimi-k2.5":     "v",
    "glm-5.1":       "D",
}


# Some level cells were measured on an alternate proxy deployment of the SAME
# underlying model. FW-GLM-5.1 is glm-5.1 behind an endpoint with a higher
# gateway timeout, used only because the default glm-5.1 endpoint dropped the
# longer level-200 connections (~450-510s cutoff). It is the same model, so its
# points belong on the glm-5.1 line rather than as a separate series.
LLM_ALIASES = {
    "FW-GLM-5.1": "glm-5.1",
}


def _parse_model(model_str: str):
    """'gpt-4o+pipeline' -> ('gpt-4o', 'pipeline')."""
    if "+" in model_str:
        llm, strategy = model_str.rsplit("+", 1)
        return LLM_ALIASES.get(llm, llm), strategy
    return LLM_ALIASES.get(model_str, model_str), "unknown"


def load_pipeline_rows(base: str, strategy_filter: str = "pipeline"):
    """Aggregate rows for one strategy: dict[(llm, level)] -> {tokens:[...], time:[...]}."""
    pattern = os.path.join(base, "run_*", "STRESS_TEST_TABLE.csv")
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No STRESS_TEST_TABLE.csv found under {base}")
    data = defaultdict(lambda: {"tokens": [], "time": []})
    for p in paths:
        with open(p, newline="") as fh:
            for row in csv.DictReader(fh):
                llm, strategy = _parse_model(row["model"])
                if strategy != strategy_filter:
                    continue
                try:
                    level = int(row["error_count"])
                except (KeyError, ValueError):
                    continue
                if level not in TARGET_LEVELS:
                    continue
                total_tok = int(row.get("total_tokens", 0) or 0)
                dur = float(row.get("duration_s", 0) or 0)
                # A run only counts if it actually produced output (total_tokens
                # > 0). Failed runs (timeout/429) report total_tokens==0 but a
                # large wall-time; including their duration would plot a timeout
                # as if it were a real completion time, so gate BOTH on tokens.
                if total_tok > 0:
                    data[(llm, level)]["tokens"].append(total_tok)
                    if dur > 0:
                        data[(llm, level)]["time"].append(dur)
    return data, paths


def _llms_present(data, exclude=None):
    exclude = set(exclude or [])
    present = {llm for (llm, _lvl) in data} - exclude
    ordered = [m for m in LLM_ORDER if m in present]
    ordered += sorted(present - set(ordered))
    return ordered


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="outputs/pipeline_stress",
                    help="directory containing run_* folders")
    ap.add_argument("--out", default=None, help="output PNG path")
    ap.add_argument("--exclude", default="", help="comma-separated LLMs to omit")
    ap.add_argument("--strategy", default="pipeline",
                    help="which strategy's rows to plot (pipeline|simple|bioguider)")
    args = ap.parse_args()

    exclude = [m.strip() for m in args.exclude.split(",") if m.strip()]
    data, paths = load_pipeline_rows(args.base, strategy_filter=args.strategy)
    llms = _llms_present(data, exclude=exclude)

    fig, (ax_tok, ax_time) = plt.subplots(1, 2, figsize=(14, 6))
    summary_rows = []

    for llm in llms:
        color = LLM_COLORS.get(llm, None)
        marker = LLM_MARKERS.get(llm, "o")
        # Build y-values over the full x-axis with NaN at levels that produced
        # no successful run. NaN makes matplotlib BREAK the line at that level
        # rather than drawing a straight segment across it — so a level a model
        # failed at (e.g. glm-5.1 @ 100 in pipeline mode) shows as a visible gap
        # instead of a misleading interpolation.
        tok_y = [float("nan")] * len(TARGET_LEVELS)
        time_y = [float("nan")] * len(TARGET_LEVELS)
        any_tok = any_time = False
        for i, lvl in enumerate(TARGET_LEVELS):
            cell = data.get((llm, lvl))
            if not cell:
                continue
            mtok = float(np.mean(cell["tokens"])) if cell["tokens"] else float("nan")
            mtime = float(np.mean(cell["time"])) if cell["time"] else float("nan")
            if cell["tokens"]:
                tok_y[i] = mtok
                any_tok = True
            if cell["time"]:
                time_y[i] = mtime
                any_time = True
            summary_rows.append((llm, lvl, mtok, mtime,
                                 len(cell["tokens"]), len(cell["time"])))
        if any_tok:
            ax_tok.plot(TARGET_LEVELS, tok_y, marker=marker, color=color, label=llm, lw=2)
        if any_time:
            ax_time.plot(TARGET_LEVELS, time_y, marker=marker, color=color, label=llm, lw=2)

    strat_label = args.strategy.capitalize()
    for ax, title, ylab in [
        (ax_tok, f"{strat_label} token usage", "total tokens (mean)"),
        (ax_time, f"{strat_label} wall-time", "seconds (mean)"),
    ]:
        ax.set_title(title)
        ax.set_xlabel("error level (errors per category)")
        ax.set_ylabel(ylab)
        ax.set_xticks(TARGET_LEVELS)
        ax.grid(alpha=0.3)
        ax.legend(title="LLM", fontsize=9)

    fig.suptitle(f"BioGuider {args.strategy} — token & time vs error level", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    out_path = args.out or os.path.join(args.base, "pipeline_tokentime.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")

    csv_path = os.path.splitext(out_path)[0] + "_summary.csv"
    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["llm", "error_level", "mean_total_tokens", "mean_time_s",
                    "n_token_runs", "n_time_runs"])
        for llm, lvl, mtok, mtime, ntok, ntime in summary_rows:
            w.writerow([llm, lvl,
                        "" if mtok != mtok else round(mtok, 1),
                        "" if mtime != mtime else round(mtime, 1),
                        ntok, ntime])

    print(f"wrote {out_path}")
    print(f"wrote {csv_path}  ({len(summary_rows)} rows from {len(paths)} run(s))")


if __name__ == "__main__":
    main()
