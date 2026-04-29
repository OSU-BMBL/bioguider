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
        if n == 0:
            return
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


# ---------------------------------------------------------------------------
# Rescored figure renderer — reads STRESS_TEST_TABLE_RESCORED.csv directly
# ---------------------------------------------------------------------------

class _RescoredPlotter:
    """Render the six benchmark figures from a RESCORED CSV (no JSON needed).

    The rescored CSV has columns: file_stem, model, error_count,
    total_injected, fixed, unfixed, fix_rate, precision, recall,
    f1_score, total_injected_scorable, fixed_scorable, f1_score_scorable,
    total_injected_content, fixed_content, f1_score_content,
    total_injected_hygiene, fixed_hygiene, f1_score_hygiene, duration_s.
    """

    def __init__(
        self,
        out_dir: PathLike,
        csv_path: PathLike,
        metric_col: str,
        suffix: str,
        category_filter: Optional[set] = None,
    ):
        self.out_dir = Path(out_dir)
        self.metric_col = metric_col
        self.suffix = suffix
        self.category_filter = category_filter
        self._rows = self._load(Path(csv_path))

    def _load(self, csv_path: Path) -> list:
        rows = []
        with csv_path.open() as fh:
            for row in csv.DictReader(fh):
                rows.append(
                    {
                        "model": row["model"],
                        "error_count": int(row["error_count"]),
                        "f1": float(row.get(self.metric_col, 0.0)),
                        "fix_rate": float(row.get("fix_rate", 0.0)),
                        "fixed": int(row.get("fixed", 0)),
                        "unfixed": int(row.get("unfixed", 0)),
                        "duration_s": float(row.get("duration_s", 0.0)),
                    }
                )
        return rows

    def _save(self, fig: plt.Figure, name: str) -> None:
        for ext in ("png", "pdf"):
            fig.savefig(self.out_dir / f"{name}.{ext}")
        plt.close(fig)

    def _label(self, base: str) -> str:
        return f"{base} ({self.metric_col})"

    def fig1(self) -> None:
        data: dict = defaultdict(dict)
        for r in self._rows:
            data[r["model"]][r["error_count"]] = r["f1"]
        fig, ax = plt.subplots(figsize=(8, 5))
        for model, pts in sorted(data.items()):
            xs = sorted(pts)
            ax.plot(xs, [pts[x] for x in xs], marker="o", label=model)
        ax.set_xlabel("Error Count")
        ax.set_ylabel(self._label("F1 Score"))
        ax.set_title(f"F1 Score by Error Level ({self.suffix})")
        ax.set_ylim(0, 1.05)
        ax.legend(loc="lower left", fontsize=8)
        self._save(fig, f"fig1_f1_by_error_level_{self.suffix}")

    def fig2(self) -> None:
        model_vals: dict = defaultdict(list)
        for r in self._rows:
            model_vals[r["model"]].append(r["f1"])
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
        ax.set_xlabel(self._label("Mean F1 Score"))
        ax.set_title(f"Average F1 Score by Model (±95 % CI) ({self.suffix})")
        ax.set_xlim(0, 1.1)
        self._save(fig, f"fig2_avg_f1_by_model_{self.suffix}")

    def fig3(self) -> None:
        """Per-category fix-rate heatmap, filtered to ``self.category_filter`` if set.

        Reads sibling ``*_CATEGORY_DETAIL.csv`` (aggregate or per-vignette depending
        on self.out_dir). Falls back to the placeholder mean-F1 bar if detail CSV
        is missing.
        """
        detail_csv = self._locate_detail_csv()
        if detail_csv is None or not detail_csv.exists():
            # Fallback: placeholder bar chart
            model_vals: dict = defaultdict(list)
            for r in self._rows:
                model_vals[r["model"]].append(r["f1"])
            models = sorted(model_vals.keys())
            means = [float(np.mean(model_vals[m])) for m in models]
            fig, ax = plt.subplots(figsize=(max(6, len(models) * 1.5), 4))
            ax.bar(models, means, color="#2563eb")
            ax.set_ylim(0, 1.1)
            ax.set_ylabel(self._label("Mean F1"))
            ax.set_title(f"Mean F1 by Model — {self.suffix} (category detail CSV missing)")
            ax.tick_params(axis="x", rotation=20)
            self._save(fig, f"fig3_category_heatmap_{self.suffix}")
            return

        acc: dict = defaultdict(lambda: defaultdict(list))
        with detail_csv.open() as fh:
            for row in csv.DictReader(fh):
                cat = row.get("category", "")
                if self.category_filter is not None and cat not in self.category_filter:
                    continue
                try:
                    fr = float(row.get("fix_rate", 0.0))
                except (TypeError, ValueError):
                    continue
                acc[row["model"]][cat].append(fr)

        if not acc:
            # Filter matched no categories — render empty placeholder
            fig, ax = plt.subplots(figsize=(6, 3))
            ax.text(0.5, 0.5, f"No {self.suffix.upper()} categories injected in this run",
                    ha="center", va="center", fontsize=11)
            ax.set_axis_off()
            self._save(fig, f"fig3_category_heatmap_{self.suffix}")
            return

        models = sorted(acc.keys())
        categories = sorted({c for m in acc for c in acc[m]})
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
        ax.set_title(f"Category Fix Rate Heatmap — {self.suffix.upper()} "
                     f"({len(categories)} categories)")
        self._save(fig, f"fig3_category_heatmap_{self.suffix}")

    def _locate_detail_csv(self):
        """Find the *_CATEGORY_DETAIL.csv sibling next to the aggregate or per-vignette CSV."""
        # out_dir is either <run>/_aggregate/ or <run>/<stem>/
        for name in ("AGGREGATE_CATEGORY_DETAIL.csv", "STRESS_TEST_CATEGORY_DETAIL.csv"):
            candidate = self.out_dir / name
            if candidate.exists():
                return candidate
        return None

    def fig4(self) -> None:
        models_sorted = sorted(
            {r["model"] for r in self._rows},
            key=lambda m: -np.mean([r["f1"] for r in self._rows if r["model"] == m]),
        )
        error_counts = sorted({r["error_count"] for r in self._rows})
        model_fr: dict = {m: {} for m in models_sorted}
        for r in self._rows:
            model_fr[r["model"]][r["error_count"]] = r["fix_rate"]
        x = np.arange(len(error_counts))
        n = len(models_sorted)
        if n == 0:
            return
        width = 0.8 / n
        offsets = np.linspace(-0.4 + width / 2, 0.4 - width / 2, n)
        fig, ax = plt.subplots(figsize=(10, 5))
        for i, model in enumerate(models_sorted):
            vals = [model_fr[model].get(ec, 0.0) for ec in error_counts]
            ax.bar(x + offsets[i], vals, width, label=model)
        ax.set_xticks(x)
        ax.set_xticklabels(error_counts)
        ax.set_xlabel("Error Count")
        ax.set_ylabel(self._label("Fix Rate"))
        ax.set_title(f"Fix Rate by Error Level and Model ({self.suffix})")
        ax.set_ylim(0, 1.1)
        ax.legend(fontsize=7)
        self._save(fig, f"fig4_fix_rate_{self.suffix}")

    def fig5(self) -> None:
        data: dict = defaultdict(dict)
        for r in self._rows:
            data[r["model"]][r["error_count"]] = r["duration_s"]
        fig, ax = plt.subplots(figsize=(8, 5))
        for model, pts in sorted(data.items()):
            xs = sorted(pts)
            ax.plot(xs, [pts[x] for x in xs], marker="s", label=model)
        ax.set_xlabel("Error Count")
        ax.set_ylabel("Duration (seconds)")
        ax.set_title(f"Response Time by Error Level ({self.suffix})")
        ax.legend(fontsize=8)
        self._save(fig, f"fig5_response_time_{self.suffix}")

    def fig6(self) -> None:
        error_counts = sorted({r["error_count"] for r in self._rows})
        median_level = error_counts[len(error_counts) // 2]
        subset = sorted(
            [r for r in self._rows if r["error_count"] == median_level],
            key=lambda r: -r["f1"],
        )
        models = [r["model"] for r in subset]
        fixed = [r["fixed"] for r in subset]
        unfixed = [r["unfixed"] for r in subset]
        x = np.arange(len(models))
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(x, fixed, label="Fixed", color="#2ecc71")
        ax.bar(x, unfixed, bottom=fixed, label="Unfixed", color="#e74c3c")
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel("Error Count")
        ax.set_title(f"Fixed vs Unfixed at Error Level {median_level} ({self.suffix})")
        ax.legend()
        self._save(fig, f"fig6_fixed_unfixed_{self.suffix}")

    def fig_heatmap(self) -> None:
        """Heatmap: rows=models, cols=error levels, cell=F1 score (metric_col)."""
        models = sorted({r["model"] for r in self._rows})
        error_counts = sorted({r["error_count"] for r in self._rows})
        if not models or not error_counts:
            return

        # Build lookup: (model, error_count) -> f1
        lookup: dict = {}
        for r in self._rows:
            lookup[(r["model"], r["error_count"])] = r["f1"]

        matrix = np.zeros((len(models), len(error_counts)))
        for i, model in enumerate(models):
            for j, ec in enumerate(error_counts):
                matrix[i, j] = lookup.get((model, ec), 0.0)

        fig, ax = plt.subplots(
            figsize=(max(6, len(error_counts) * 0.9), max(3, len(models) * 0.7))
        )
        im = ax.imshow(matrix, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(np.arange(len(error_counts)))
        ax.set_yticks(np.arange(len(models)))
        ax.set_xticklabels(error_counts, fontsize=8)
        ax.set_yticklabels(models, fontsize=8)
        ax.set_xlabel("Error Count")
        ax.set_ylabel("Model")
        for i in range(len(models)):
            for j in range(len(error_counts)):
                color = "black" if 0.25 < matrix[i, j] < 0.8 else "white"
                ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center",
                        fontsize=7, color=color)
        plt.colorbar(im, ax=ax, label=self._label("F1 Score"))
        ax.set_title(f"F1 Score Heatmap ({self.suffix})")
        self._save(fig, f"heatmap_{self.suffix}")

    def fig_skill_comparison(self, csv_path: PathLike, suffix: str = "skill") -> None:
        """Grouped bar chart: x=error_count, bars=skill, y=F1 scorable.

        Reads *csv_path* directly (must have a ``skill`` column plus the
        standard columns written by ``_write_skill_comparison_csv``).
        Saves ``fig_skill_comparison_{suffix}.{png,pdf}`` into ``self.out_dir``.
        """
        csv_path = Path(csv_path)
        rows: list = []
        with csv_path.open() as fh:
            for row in csv.DictReader(fh):
                try:
                    rows.append(
                        {
                            "skill": row["skill"],
                            "error_count": int(row["error_count"]),
                            "f1_scorable": float(row.get("f1_score_scorable", 0.0)),
                        }
                    )
                except (KeyError, ValueError):
                    continue

        if not rows:
            return

        skills = sorted({r["skill"] for r in rows})
        error_counts = sorted({r["error_count"] for r in rows})

        from collections import defaultdict as _dd

        skill_data: dict = {s: _dd(list) for s in skills}
        for r in rows:
            skill_data[r["skill"]][r["error_count"]].append(r["f1_scorable"])

        x = np.arange(len(error_counts))
        n = len(skills)
        width = 0.8 / n
        offsets = np.linspace(-0.4 + width / 2, 0.4 - width / 2, n)

        fig, ax = plt.subplots(figsize=(max(6, len(error_counts) * 1.2), 5))
        for i, skill in enumerate(skills):
            vals = [
                float(np.mean(skill_data[skill][ec])) if skill_data[skill][ec] else 0.0
                for ec in error_counts
            ]
            ax.bar(x + offsets[i], vals, width, label=skill)

        ax.set_xticks(x)
        ax.set_xticklabels(error_counts)
        ax.set_xlabel("Error Count")
        ax.set_ylabel("F1 Score (scorable)")
        ax.set_title(f"Skill Comparison — F1 Scorable by Error Level ({suffix})")
        ax.set_ylim(0, 1.1)
        ax.legend(fontsize=8)
        self._save(fig, f"fig_skill_comparison_{suffix}")

    def render_all(self) -> None:
        self.fig1()
        self.fig2()
        self.fig3()
        self.fig4()
        self.fig5()
        self.fig6()
        self.fig_heatmap()


def render_rescored_figures(
    run_dir: PathLike,
    metric_col: str = "f1_score_content",
    suffix: str = "content",
) -> None:
    """Render six rescored figures for *run_dir* using *metric_col* as the F1 signal.

    Reads ``STRESS_TEST_TABLE_RESCORED.csv`` from each per-vignette subdir and
    ``AGGREGATE_TABLE_RESCORED.csv`` from ``_aggregate/``. Writes
    ``fig{1..6}_<suffix>.{png,pdf}`` into each vignette dir and the aggregate dir.
    Original ``fig{1..6}.{png,pdf}`` are never touched.

    When ``suffix in {"content", "hygiene"}``, fig3 filters the category heatmap to
    the matching ``CONTENT_CATEGORIES`` or ``HYGIENE_CATEGORIES`` set from
    ``bioguider.managers.config``.
    """
    from bioguider.managers.config import CONTENT_CATEGORIES, HYGIENE_CATEGORIES

    category_filter = None
    if suffix == "content":
        category_filter = set(CONTENT_CATEGORIES)
    elif suffix == "hygiene":
        category_filter = set(HYGIENE_CATEGORIES)

    run_dir = Path(run_dir)
    agg_csv = run_dir / "_aggregate" / "AGGREGATE_TABLE_RESCORED.csv"

    stem_dirs = sorted(
        d for d in run_dir.iterdir()
        if d.is_dir() and d.name != "_aggregate"
    )

    for stem_dir in stem_dirs:
        stem_csv = stem_dir / "STRESS_TEST_TABLE_RESCORED.csv"
        if not stem_csv.exists():
            continue
        plotter = _RescoredPlotter(
            stem_dir, stem_csv, metric_col, suffix, category_filter=category_filter
        )
        plotter.render_all()

    if agg_csv.exists():
        agg_plotter = _RescoredPlotter(
            run_dir / "_aggregate", agg_csv, metric_col, suffix,
            category_filter=category_filter,
        )
        agg_plotter.render_all()


def render_heatmap(
    csv_path: PathLike,
    out_dir: PathLike,
    metric_col: str = "f1_score",
    suffix: str = "heatmap",
) -> None:
    """Render an F1 heatmap (rows=models, cols=error levels) from *csv_path*.

    Writes ``heatmap_{suffix}.{png,pdf}`` into *out_dir*.

    Args:
        csv_path: Path to a STRESS_TEST_TABLE*.csv (or any rescored variant)
                  that contains columns ``model``, ``error_count``, and
                  *metric_col*.
        out_dir:  Directory where the output images are saved.
        metric_col: Column to use as the cell value (default ``f1_score``).
        suffix:   File-name suffix for the output images.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plotter = _RescoredPlotter(out_dir, csv_path, metric_col, suffix)
    plotter.fig_heatmap()


def render_skill_comparison(
    csv_path: PathLike,
    out_dir: PathLike,
    suffix: str = "skill",
) -> None:
    """Render a grouped bar chart comparing skills from a SKILL_COMPARISON or
    SKILL_MATRIX_TABLE CSV (must contain a ``skill`` column).

    Detects the ``skill`` column automatically.  Writes
    ``fig_skill_comparison_{suffix}.{png,pdf}`` into *out_dir*.

    Args:
        csv_path: Path to SKILL_COMPARISON.csv or SKILL_MATRIX_TABLE.csv.
        out_dir:  Directory where output images are saved.
        suffix:   File-name suffix appended to ``fig_skill_comparison_``.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # _RescoredPlotter needs a valid metric_col for its _load(); we pass a
    # dummy placeholder and rely only on fig_skill_comparison which reads
    # csv_path directly without going through _load().
    plotter = _RescoredPlotter(out_dir, csv_path, "f1_score", suffix)
    plotter.fig_skill_comparison(csv_path, suffix=suffix)
