"""Citation correlation analysis for Single-cell software subset.

Computes Pearson/Spearman correlations between documentation scores and
citation metrics (citation_per_year, GitHub stars). Fits both linear and
exponential models per the April 24 meeting decision to focus on Single-cell
software and try exponential fitting if linear shows no correlation.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any, Optional

import numpy as np

try:
    from scipy import stats as sp_stats
    from scipy.optimize import curve_fit

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAS_MPL = True
except ImportError:
    HAS_MPL = False


def _exp_model(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    return a * np.exp(b * x) + c


def compute_correlations(
    doc_stats_csv: Path,
    citation_csv: Path,
    domain_filter: str = "single_cell",
    score_columns: Optional[list[str]] = None,
    citation_column: str = "citation_per_year",
) -> list[dict[str, Any]]:
    """Compute correlation between doc scores and citation metrics.

    Args:
        doc_stats_csv: CSV from doc_stats.save_doc_stats().
        citation_csv: CSV with columns: software, domain, citation_per_year,
                      github_stars (user-provided).
        domain_filter: Only analyze software in this domain.
        score_columns: Which score columns to correlate. Defaults to the 4 categories.
        citation_column: Which citation metric to use.

    Returns:
        List of correlation result dicts.
    """
    if not HAS_SCIPY:
        raise ImportError("scipy is required for citation analysis")

    if score_columns is None:
        score_columns = [
            "readme_score",
            "installation_score",
            "userguide_score",
            "tutorial_score",
        ]

    doc_data = {}
    with doc_stats_csv.open() as f:
        for row in csv.DictReader(f):
            doc_data[row["software"]] = row

    citations = {}
    with citation_csv.open() as f:
        for row in csv.DictReader(f):
            if domain_filter and row.get("domain", "").lower() != domain_filter.lower():
                continue
            citations[row["software"]] = row

    merged_keys = set(doc_data) & set(citations)
    if len(merged_keys) < 5:
        return [{"error": f"Too few samples after merge: {len(merged_keys)}"}]

    results = []
    for col in score_columns:
        xs, ys = [], []
        for key in sorted(merged_keys):
            score_val = float(doc_data[key].get(col, 0))
            cite_val = float(citations[key].get(citation_column, 0))
            if score_val > 0 and cite_val > 0:
                xs.append(score_val)
                ys.append(cite_val)

        if len(xs) < 5:
            results.append({
                "score_column": col,
                "citation_column": citation_column,
                "n": len(xs),
                "error": "too_few_samples",
            })
            continue

        x_arr = np.array(xs)
        y_arr = np.array(ys)

        pearson_r, pearson_p = sp_stats.pearsonr(x_arr, y_arr)
        spearman_r, spearman_p = sp_stats.spearmanr(x_arr, y_arr)

        slope, intercept, r_val, p_val, std_err = sp_stats.linregress(x_arr, y_arr)
        linear_r2 = r_val ** 2

        exp_r2 = None
        exp_params = None
        try:
            popt, _ = curve_fit(
                _exp_model,
                x_arr,
                y_arr,
                p0=[1, 0.01, 0],
                maxfev=5000,
            )
            y_pred = _exp_model(x_arr, *popt)
            ss_res = np.sum((y_arr - y_pred) ** 2)
            ss_tot = np.sum((y_arr - np.mean(y_arr)) ** 2)
            exp_r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
            exp_params = {"a": popt[0], "b": popt[1], "c": popt[2]}
        except (RuntimeError, ValueError):
            pass

        results.append({
            "score_column": col,
            "citation_column": citation_column,
            "n": len(xs),
            "pearson_r": round(pearson_r, 4),
            "pearson_p": round(pearson_p, 4),
            "spearman_r": round(spearman_r, 4),
            "spearman_p": round(spearman_p, 4),
            "linear_r2": round(linear_r2, 4),
            "linear_slope": round(slope, 4),
            "linear_intercept": round(intercept, 4),
            "linear_p": round(p_val, 4),
            "exp_r2": round(exp_r2, 4) if exp_r2 is not None else None,
            "exp_params": exp_params,
            "best_fit": "exponential" if exp_r2 and exp_r2 > linear_r2 else "linear",
        })

    return results


def save_correlations(results: list[dict[str, Any]], output_path: Path) -> Path:
    """Write correlation results to CSV."""
    if not results:
        return output_path

    fieldnames = [
        "score_column", "citation_column", "n",
        "pearson_r", "pearson_p", "spearman_r", "spearman_p",
        "linear_r2", "linear_slope", "linear_intercept", "linear_p",
        "exp_r2", "best_fit",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)

    return output_path


def plot_scatter(
    doc_stats_csv: Path,
    citation_csv: Path,
    output_dir: Path,
    domain_filter: str = "single_cell",
    citation_column: str = "citation_per_year",
) -> list[Path]:
    """Generate scatter plots for each score vs citation metric.

    Returns list of saved file paths.
    """
    if not HAS_MPL or not HAS_SCIPY:
        return []

    score_columns = [
        "readme_score", "installation_score",
        "userguide_score", "tutorial_score",
    ]

    doc_data = {}
    with doc_stats_csv.open() as f:
        for row in csv.DictReader(f):
            doc_data[row["software"]] = row

    citations = {}
    with citation_csv.open() as f:
        for row in csv.DictReader(f):
            if domain_filter and row.get("domain", "").lower() != domain_filter.lower():
                continue
            citations[row["software"]] = row

    merged_keys = set(doc_data) & set(citations)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved = []

    for col in score_columns:
        xs, ys, labels = [], [], []
        for key in sorted(merged_keys):
            score_val = float(doc_data[key].get(col, 0))
            cite_val = float(citations[key].get(citation_column, 0))
            if score_val > 0 and cite_val > 0:
                xs.append(score_val)
                ys.append(cite_val)
                labels.append(key)

        if len(xs) < 5:
            continue

        x_arr = np.array(xs)
        y_arr = np.array(ys)

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter(x_arr, y_arr, alpha=0.7, edgecolors="k", linewidths=0.5)

        slope, intercept, r_val, p_val, _ = sp_stats.linregress(x_arr, y_arr)
        x_fit = np.linspace(x_arr.min(), x_arr.max(), 100)
        ax.plot(x_fit, slope * x_fit + intercept, "r--", linewidth=1.5,
                label=f"Linear (R²={r_val**2:.3f}, p={p_val:.3f})")

        try:
            popt, _ = curve_fit(_exp_model, x_arr, y_arr, p0=[1, 0.01, 0], maxfev=5000)
            y_exp = _exp_model(x_fit, *popt)
            ss_res = np.sum((y_arr - _exp_model(x_arr, *popt)) ** 2)
            ss_tot = np.sum((y_arr - np.mean(y_arr)) ** 2)
            exp_r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
            ax.plot(x_fit, y_exp, "b-", linewidth=1.5,
                    label=f"Exponential (R²={exp_r2:.3f})")
        except (RuntimeError, ValueError):
            pass

        ax.set_xlabel(col.replace("_", " ").title())
        ax.set_ylabel(citation_column.replace("_", " ").title())
        ax.set_title(f"{col.replace('_', ' ').title()} vs {citation_column.replace('_', ' ').title()}\n(Single-cell, n={len(xs)})")
        ax.legend(fontsize=8)

        for ext in ("png", "pdf"):
            path = output_dir / f"citation_{col}.{ext}"
            fig.savefig(path, dpi=150, bbox_inches="tight")
            saved.append(path)
        plt.close(fig)

    return saved
