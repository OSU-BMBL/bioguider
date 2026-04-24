"""Benchmark result visualiser — consumes STRESS_TEST_RESULTS.json +
STRESS_TEST_CATEGORY_DETAIL.csv and emits fig{1..6}.{png,pdf}."""
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Union

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from typing import Optional

mpl.rcParams.update(
    {
        "figure.dpi": 150,
        "savefig.bbox": "tight",
        "font.family": "DejaVu Sans",
    }
)

PathLike = Union[str, Path]


def _f1(r: dict) -> float:
    """Prefer scorable F1 (UNSCORABLE_CATEGORIES excluded) when present.

    Legacy STRESS_TEST_RESULTS.json written pre-D2 only has ``f1_score``.
    New runs carry both ``f1_score`` (headline) and ``f1_score_scorable``
    (the paper-figure headline). Plotter uses scorable where available so
    the ``function`` category doesn't inflate results.
    """
    return r.get("f1_score_scorable", r.get("f1_score", 0.0))


def _fix_rate(r: dict) -> float:
    return r.get("fix_rate_scorable", r.get("fix_rate", 0.0))


class BenchmarkPlotter:
    """Load benchmark artifacts from *out_dir* and render all six figures."""

    def __init__(self, out_dir: PathLike):
        self.out_dir = Path(out_dir)
        self._results = self._load_results()
        self._cat_rows = self._load_category_detail()
        # Title annotation — drop when no scorable numbers present.
        self._has_scorable = any("f1_score_scorable" in r for r in self._results)
        self._suffix = " — scorable (function excluded)" if self._has_scorable else ""

    # ------------------------------------------------------------------
    # Loaders
    # ------------------------------------------------------------------

    def _load_results(self) -> list:
        with open(self.out_dir / "STRESS_TEST_RESULTS.json") as fh:
            return json.load(fh)["results"]

    def _load_category_detail(self) -> list:
        path = self.out_dir / "STRESS_TEST_CATEGORY_DETAIL.csv"
        rows = []
        with open(path) as fh:
            for row in csv.DictReader(fh):
                rows.append(
                    {
                        "model": row["model"],
                        "error_level": int(row["error_level"]),
                        "category": row["category"],
                        "injected": int(row["injected"]),
                        "fixed": int(row["fixed"]),
                        "unfixed": int(row["unfixed"]),
                        "fix_rate": float(row["fix_rate"]),
                    }
                )
        return rows

    # ------------------------------------------------------------------
    # Save helper
    # ------------------------------------------------------------------

    def _save(self, fig: plt.Figure, name: str) -> None:
        for ext in ("png", "pdf"):
            fig.savefig(self.out_dir / f"{name}.{ext}")
        plt.close(fig)

    # ------------------------------------------------------------------
    # Figures
    # ------------------------------------------------------------------

    def fig1_f1_by_error_level(self) -> None:
        """Line plot: x=error_count, y=F1, one line per model."""
        data: dict = defaultdict(dict)
        for r in self._results:
            data[r["model"]][r["error_count"]] = _f1(r)

        fig, ax = plt.subplots(figsize=(8, 5))
        for model, pts in sorted(data.items()):
            xs = sorted(pts)
            ax.plot(xs, [pts[x] for x in xs], marker="o", label=model)
        ax.set_xlabel("Error Count")
        ax.set_ylabel("F1 Score" + (" (scorable)" if self._has_scorable else ""))
        ax.set_title("F1 Score by Error Level" + self._suffix)
        ax.set_ylim(0, 1.05)
        ax.legend(loc="lower left", fontsize=8)
        self._save(fig, "fig1_f1_by_error_level")

    def fig2_avg_f1_by_model(self) -> None:
        """Horizontal bar: mean ± 95 % CI of F1 across error levels, sorted desc."""
        model_vals: dict = defaultdict(list)
        for r in self._results:
            model_vals[r["model"]].append(_f1(r))

        models = sorted(model_vals, key=lambda m: -np.mean(model_vals[m]))
        means = [float(np.mean(model_vals[m])) for m in models]
        ci95 = [
            1.96 * float(np.std(model_vals[m])) / np.sqrt(len(model_vals[m]))
            if len(model_vals[m]) > 1
            else 0.0
            for m in models
        ]

        fig, ax = plt.subplots(figsize=(7, max(3, len(models) * 0.6)))
        y_pos = np.arange(len(models))
        ax.barh(y_pos, means, xerr=ci95, align="center", height=0.6, capsize=4)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(models, fontsize=9)
        ax.set_xlabel("Mean F1 Score" + (" (scorable)" if self._has_scorable else ""))
        ax.set_title("Average F1 Score by Model (±95 % CI)" + self._suffix)
        ax.set_xlim(0, 1.1)
        self._save(fig, "fig2_avg_f1_by_model")

    def fig3_category_heatmap(self) -> None:
        """Heatmap: rows=models, cols=categories, cell=mean fix_rate."""
        acc: dict = defaultdict(lambda: defaultdict(list))
        for row in self._cat_rows:
            acc[row["model"]][row["category"]].append(row["fix_rate"])

        models = sorted(acc.keys())
        categories = sorted({r["category"] for r in self._cat_rows})
        matrix = np.zeros((len(models), len(categories)))
        for i, model in enumerate(models):
            for j, cat in enumerate(categories):
                vals = acc[model].get(cat, [])
                matrix[i, j] = float(np.mean(vals)) if vals else 0.0

        fig, ax = plt.subplots(
            figsize=(max(8, len(categories) * 0.9), max(3, len(models) * 0.7))
        )
        im = ax.imshow(matrix, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(np.arange(len(categories)))
        ax.set_yticks(np.arange(len(models)))
        ax.set_xticklabels(categories, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(models, fontsize=8)
        for i in range(len(models)):
            for j in range(len(categories)):
                color = "black" if 0.25 < matrix[i, j] < 0.8 else "white"
                ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center",
                        fontsize=6, color=color)
        plt.colorbar(im, ax=ax, label="Fix Rate")
        ax.set_title("Category Fix Rate Heatmap")
        self._save(fig, "fig3_category_heatmap")

    def fig4_fix_rate(self) -> None:
        """Grouped bar: x=error_count, hue=model, sorted by mean F1 desc."""
        models_sorted = sorted(
            {r["model"] for r in self._results},
            key=lambda m: -np.mean([_f1(r) for r in self._results if r["model"] == m]),
        )
        error_counts = sorted({r["error_count"] for r in self._results})
        model_fr: dict = {m: {} for m in models_sorted}
        for r in self._results:
            model_fr[r["model"]][r["error_count"]] = _fix_rate(r)

        x = np.arange(len(error_counts))
        n = len(models_sorted)
        width = 0.8 / n
        offsets = np.linspace(-0.4 + width / 2, 0.4 - width / 2, n)

        fig, ax = plt.subplots(figsize=(10, 5))
        for i, model in enumerate(models_sorted):
            vals = [model_fr[model].get(ec, 0.0) for ec in error_counts]
            ax.bar(x + offsets[i], vals, width, label=model)
        ax.set_xticks(x)
        ax.set_xticklabels(error_counts)
        ax.set_xlabel("Error Count")
        ax.set_ylabel("Fix Rate" + (" (scorable)" if self._has_scorable else ""))
        ax.set_title("Fix Rate by Error Level and Model" + self._suffix)
        ax.set_ylim(0, 1.1)
        ax.legend(fontsize=7)
        self._save(fig, "fig4_fix_rate")

    def fig5_response_time(self) -> None:
        """Line plot: duration_seconds vs error_count per model."""
        data: dict = defaultdict(dict)
        for r in self._results:
            data[r["model"]][r["error_count"]] = r["duration_seconds"]

        fig, ax = plt.subplots(figsize=(8, 5))
        for model, pts in sorted(data.items()):
            xs = sorted(pts)
            ax.plot(xs, [pts[x] for x in xs], marker="s", label=model)
        ax.set_xlabel("Error Count")
        ax.set_ylabel("Duration (seconds)")
        ax.set_title("Response Time by Error Level")
        ax.legend(fontsize=8)
        self._save(fig, "fig5_response_time")

    def fig6_fixed_unfixed(self) -> None:
        """Stacked bar (fixed/unfixed) per model at the median error level."""
        error_counts = sorted({r["error_count"] for r in self._results})
        median_level = error_counts[len(error_counts) // 2]

        subset = sorted(
            [r for r in self._results if r["error_count"] == median_level],
            key=lambda r: -_f1(r),
        )
        models = [r["model"] for r in subset]
        fixed = [r["errors_fixed"] for r in subset]
        unfixed = [r["errors_unfixed"] for r in subset]

        x = np.arange(len(models))
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(x, fixed, label="Fixed", color="#2ecc71")
        ax.bar(x, unfixed, bottom=fixed, label="Unfixed", color="#e74c3c")
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel("Error Count")
        ax.set_title(f"Fixed vs Unfixed at Error Level {median_level}")
        ax.legend()
        self._save(fig, "fig6_fixed_unfixed")

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    def render_all(self, out_dir: Optional[PathLike] = None) -> None:
        """Render all six figures to *out_dir* (defaults to constructor path)."""
        if out_dir is not None:
            self.out_dir = Path(out_dir)
        self.fig1_f1_by_error_level()
        self.fig2_avg_f1_by_model()
        self.fig3_category_heatmap()
        self.fig4_fix_rate()
        self.fig5_response_time()
        self.fig6_fixed_unfixed()
