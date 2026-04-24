"""Render benchmark figure mockups with synthetic numbers.

Qin sketched the target figure on the whiteboard on 2026-04-23:

  - Main panel  — F1 bar chart across open-source LLMs + Claude-Code agent
                  at a fixed total-error budget (300), with Model A highlighted.
  - Gradient    — F1_scorable vs total errors (50 / 100 / 200 / 300).
  - Thinking vs general mode — F1, wall-clock, token usage side-by-side.
  - Methods callout — 4-stage unified prompt + code-consistency differentiator.

This script produces three PNG mockups using dummy numbers so the layout can
be reviewed BEFORE real results land. Once Qin picks the default model and
benchmark runs complete, swap the ``_synthetic_*`` loaders for real parsers
of STRESS_TEST_RESULTS.json / BENCHMARK_RESULTS.json.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

mpl.rcParams.update(
    {
        "figure.dpi": 150,
        "savefig.bbox": "tight",
        "font.family": "DejaVu Sans",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titleweight": "bold",
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "legend.frameon": False,
    }
)

OUT = Path(__file__).resolve().parents[1] / "outputs" / "figure_mockups"
OUT.mkdir(parents=True, exist_ok=True)


# ----- synthetic data (replace with real parsers) ----------------------

MODELS = ["Kimi (A)", "GLM", "GPT-OSS", "Qwen", "DeepSeek", "CC Coder"]
MODEL_COLORS = ["#2E86AB", "#5DA5DA", "#5DA5DA", "#5DA5DA", "#5DA5DA", "#B3423E"]


def _synthetic_model_f1():
    # F1_scorable (headline) and F1_all (with "function" unscorables inflating).
    rng = np.random.default_rng(7)
    f1_scorable = np.array([0.71, 0.64, 0.60, 0.58, 0.55, 0.48])
    f1_all = f1_scorable + rng.uniform(0.04, 0.09, size=len(f1_scorable))
    return f1_scorable, f1_all


def _synthetic_gradient():
    # One line per system; dummy sigmoid-like decay of F1 as errors scale.
    xs = np.array([50, 100, 200, 300])
    bioguider = np.array([0.78, 0.74, 0.71, 0.68])
    cc_coder = np.array([0.62, 0.56, 0.50, 0.45])
    return xs, bioguider, cc_coder


def _synthetic_thinking_vs_general():
    # (F1, seconds/file, k-tokens/file) for thinking vs general mode.
    thinking = {"f1": 0.71, "time_s": 38.0, "tokens_k": 14.2}
    general = {"f1": 0.66, "time_s": 11.0, "tokens_k": 5.8}
    return thinking, general


def _synthetic_category_moat():
    groups = ["text", "structure", "code", "biology", "cli_config", "prose_code_consistency"]
    bioguider = np.array([0.81, 0.78, 0.72, 0.65, 0.70, 0.67])
    cc_coder = np.array([0.78, 0.74, 0.69, 0.58, 0.55, 0.09])
    return groups, bioguider, cc_coder


# ----- renderers -------------------------------------------------------

def _panel_model_f1(ax, highlight="Kimi (A)"):
    f1_scorable, f1_all = _synthetic_model_f1()
    x = np.arange(len(MODELS))
    bar_width = 0.36
    ax.bar(
        x - bar_width / 2,
        f1_scorable,
        width=bar_width,
        color=[c if m == highlight else "#9FBCD0" for m, c in zip(MODELS, MODEL_COLORS)],
        label="F1 (scorable)",
        edgecolor="white",
        linewidth=0.6,
    )
    ax.bar(
        x + bar_width / 2,
        f1_all,
        width=bar_width,
        color="#DDDDDD",
        label="F1 (headline, incl. function)",
        edgecolor="white",
        linewidth=0.6,
        hatch="//",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(MODELS, rotation=20, ha="right")
    ax.set_ylabel("F1 score")
    ax.set_title("A  Model comparison @ 300 scorable errors", pad=14)
    ax.set_ylim(0, 1.0)
    ax.axhline(0, color="black", linewidth=0.4)
    ax.legend(loc="lower left", ncol=2, fontsize=7)
    # "Default" annotation under Kimi bar
    ax.annotate(
        "default",
        xy=(0, 0),
        xytext=(0, -0.08),
        textcoords="data",
        ha="center",
        va="top",
        fontsize=7,
        color="#2E86AB",
    )


def _panel_gradient(ax):
    xs, bioguider, cc = _synthetic_gradient()
    ax.plot(xs, bioguider, marker="o", color="#2E86AB", linewidth=2.0, label="BioGuider + Kimi")
    ax.plot(xs, cc, marker="s", color="#B3423E", linewidth=2.0, linestyle="--", label="Claude-Code Agent")
    ax.set_xticks(xs)
    ax.set_xlabel("Total scorable errors injected")
    ax.set_ylabel("F1 score (scorable)")
    ax.set_title("B  F1 vs error count  (50 / 100 / 200 / 300)")
    ax.set_ylim(0.35, 0.85)
    ax.grid(alpha=0.25, linewidth=0.5)
    ax.legend(loc="upper right")


def _panel_thinking_vs_general(ax):
    thinking, general = _synthetic_thinking_vs_general()
    metrics = ["F1", "sec/file", "k-tok/file"]
    raw_t = np.array([thinking["f1"], thinking["time_s"], thinking["tokens_k"]])
    raw_g = np.array([general["f1"], general["time_s"], general["tokens_k"]])
    # Normalise each metric to [0, 1] so F1 doesn't disappear next to seconds.
    denom = np.maximum(raw_t, raw_g)
    norm_t = raw_t / denom
    norm_g = raw_g / denom
    x = np.arange(len(metrics))
    bw = 0.35
    ax.bar(x - bw / 2, norm_t, bw, color="#2E86AB", label="Thinking mode")
    ax.bar(x + bw / 2, norm_g, bw, color="#F6AE2D", label="General mode")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_title("C  Thinking vs general mode (Kimi) — normalised per metric", pad=12)
    ax.set_ylabel("Normalised value (max per metric = 1)")
    ax.set_ylim(0, 1.25)
    ax.legend(loc="upper right", fontsize=7)
    # Print the raw numbers above each bar so the normalised view stays honest.
    for i, (vt, vg) in enumerate(zip(raw_t, raw_g)):
        ax.text(i - bw / 2, norm_t[i] + 0.02, f"{vt:g}", ha="center", va="bottom", fontsize=7)
        ax.text(i + bw / 2, norm_g[i] + 0.02, f"{vg:g}", ha="center", va="bottom", fontsize=7)


def _panel_category_moat(ax):
    groups, bioguider, cc = _synthetic_category_moat()
    y = np.arange(len(groups))
    bh = 0.38
    ax.barh(y + bh / 2, bioguider, bh, color="#2E86AB", label="BioGuider + Kimi")
    ax.barh(y - bh / 2, cc, bh, color="#B3423E", label="Claude-Code Agent")
    ax.set_yticks(y)
    ax.set_yticklabels(groups)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Fix rate")
    ax.set_title("D  Category-level fix rate (the moat)", pad=10)
    # Highlight the prose_code_consistency row as BioGuider's differentiator
    ax.axhspan(y[-1] - 0.5, y[-1] + 0.5, color="#FFEED9", alpha=0.7, zorder=0)
    # Legend in the lower-left (empty since Claude-Code bars collapse on the
    # top row, so no bar collision down there).
    ax.legend(loc="lower right", fontsize=7)
    # Annotation parked in the free region to the right of the collapsed
    # Claude-Code bar on the top (moat) row — no overlap with any other row.
    ax.annotate(
        "BioGuider scans repo code\nas authority; Claude-Code\ndoesn't see the code.",
        xy=(cc[-1] + 0.01, y[-1] + bh / 2),
        xytext=(0.22, y[-1] + 0.15),
        fontsize=7,
        ha="left",
        va="center",
        arrowprops=dict(arrowstyle="->", color="#555", lw=0.7),
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#DDD"),
    )


def _methods_callout(fig):
    # A text box docking at the bottom describing the 4-stage unified prompt.
    fig.text(
        0.5,
        -0.01,
        "Unified 4-stage prompt (identical text on both systems):   "
        "load  →  goal  →  evaluation  →  correction(n_errors, n_categories)   |   "
        "Code-consistency injection requires a code-block anchor "
        "(skip + record when missing).",
        ha="center",
        va="top",
        fontsize=8,
        color="#333",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#F2F4F7", edgecolor="#CCC"),
    )


# ----- figure variants -------------------------------------------------


def fig_2x2_main():
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
    _panel_model_f1(axes[0, 0])
    _panel_gradient(axes[0, 1])
    _panel_thinking_vs_general(axes[1, 0])
    _panel_category_moat(axes[1, 1])
    fig.suptitle("BioGuider benchmark (mockup — synthetic numbers)", y=1.02, fontsize=11)
    _methods_callout(fig)
    fig.tight_layout()
    out = OUT / "fig1_main_2x2.png"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)
    return out


def fig_compact_row():
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.6))
    _panel_model_f1(axes[0])
    _panel_gradient(axes[1])
    _panel_thinking_vs_general(axes[2])
    _panel_category_moat(axes[3])
    fig.suptitle("BioGuider benchmark — compact single-row variant", y=1.06, fontsize=11)
    _methods_callout(fig)
    fig.tight_layout()
    out = OUT / "fig2_compact_row.png"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)
    return out


def fig_moat_focus():
    """Zoom on the prose-code-consistency differentiator only."""
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    _panel_category_moat(ax)
    ax.set_title("Per-category fix rate — BioGuider vs Claude-Code Agent")
    fig.text(
        0.5,
        -0.02,
        "The prose_code_consistency group contains errors whose ground truth lives in "
        "the repo's code (package version, stat-test call, marker gene, hyperparameter).\n"
        "BioGuider scans code as authority; the Claude-Code Agent doesn't, so its fix rate "
        "collapses on exactly this group.",
        ha="center",
        va="top",
        fontsize=8,
        color="#333",
    )
    fig.tight_layout()
    out = OUT / "fig3_moat_focus.png"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)
    return out


if __name__ == "__main__":
    paths = [fig_2x2_main(), fig_compact_row(), fig_moat_focus()]
    for p in paths:
        print("wrote", p)
