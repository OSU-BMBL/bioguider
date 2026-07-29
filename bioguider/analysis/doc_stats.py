"""Document length statistics for post-hoc scoring bias correction.

Counts user guides, tutorials, and total word count per software package,
then computes length-weighted score adjustments to correct the "write more,
score lower" bias identified in the April 24 meeting.
"""

from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path
from typing import Any


def _word_count(text: str) -> int:
    return len(re.findall(r"\S+", text))


def _count_files(repo_path: Path, patterns: list[str]) -> int:
    count = 0
    for pat in patterns:
        count += len(list(repo_path.glob(pat)))
    return count


def compute_doc_stats(
    evaluation_json: Path,
    repos_root: Path,
) -> list[dict[str, Any]]:
    """Compute document length statistics for all software in an evaluation.

    Args:
        evaluation_json: Path to the evaluation results JSON (from EvaluationManager).
        repos_root: Root directory containing cloned repos (e.g. data/.adalflow/repos/).

    Returns:
        List of dicts with per-software statistics.
    """
    with evaluation_json.open() as f:
        data = json.load(f)

    results = []
    for entry in data if isinstance(data, list) else [data]:
        software = entry.get("software", entry.get("repo_name", "unknown"))
        repo_key = entry.get("repo_key", software)
        repo_path = repos_root / repo_key.replace("/", "_")

        if not repo_path.exists():
            continue

        total_words = 0
        for md_file in repo_path.rglob("*.md"):
            total_words += _word_count(md_file.read_text(errors="replace"))
        for rmd_file in repo_path.rglob("*.Rmd"):
            total_words += _word_count(rmd_file.read_text(errors="replace"))

        num_userguides = _count_files(repo_path, ["docs/*.md", "USERGUIDE.md", "GUIDE.md"])
        num_tutorials = _count_files(
            repo_path,
            ["vignettes/*.Rmd", "vignettes/*.md", "tutorials/*.md", "tutorials/*.Rmd"],
        )

        scores = entry.get("scores", {})
        results.append({
            "software": software,
            "repo_key": repo_key,
            "num_userguides": num_userguides,
            "num_tutorials": num_tutorials,
            "total_words": total_words,
            "readme_score": scores.get("readme", 0),
            "installation_score": scores.get("installation", 0),
            "userguide_score": scores.get("userguide", 0),
            "tutorial_score": scores.get("tutorial", 0),
        })

    return results


def apply_length_weighting(
    stats: list[dict[str, Any]],
    alpha: float = 0.5,
) -> list[dict[str, Any]]:
    """Add length-weighted score adjustments.

    Uses log-normalization: adjusted = raw_score * log2(1 + total_words/1000)^alpha
    to reduce the penalty on longer documents.
    """
    for row in stats:
        words = row.get("total_words", 0)
        weight = math.log2(1 + words / 1000) ** alpha if words > 0 else 1.0
        for key in ("readme_score", "installation_score", "userguide_score", "tutorial_score"):
            raw = row.get(key, 0)
            row[f"{key}_adjusted"] = round(raw * weight, 2) if raw else 0

    return stats


def save_doc_stats(stats: list[dict[str, Any]], output_path: Path) -> Path:
    """Write doc stats to CSV."""
    if not stats:
        return output_path

    fieldnames = list(stats[0].keys())
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(stats)

    return output_path
