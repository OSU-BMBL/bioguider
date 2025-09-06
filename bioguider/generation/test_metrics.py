from __future__ import annotations

import json
import re
from difflib import SequenceMatcher
from typing import Dict, Any, List, Tuple


def _lev(a: str, b: str) -> float:
    return 1.0 - SequenceMatcher(None, a, b).ratio()


def _count_markdown_issues(text: str) -> int:
    issues = 0
    # naive checks
    issues += text.count("[![") - text.count("](")  # unbalanced badge syntax
    issues += text.count("[ ")  # bad link spacing
    issues += len(re.findall(r"^#[^#\s]", text, flags=re.M))  # malformed header
    return max(0, issues)


def evaluate_fixes(baseline: str, corrupted: str, revised: str, injection_manifest: Dict[str, Any]) -> Dict[str, Any]:
    per_error: List[Dict[str, Any]] = []
    per_cat: Dict[str, Dict[str, int]] = {}

    def mark(cat: str, key: str):
        per_cat.setdefault(cat, {"total": 0, "fixed_to_baseline": 0, "fixed_to_valid": 0, "unchanged": 0, "worsened": 0})
        per_cat[cat][key] += 1

    for e in injection_manifest.get("errors", []):
        cat = e.get("category", "unknown")
        per_cat.setdefault(cat, {"total": 0, "fixed_to_baseline": 0, "fixed_to_valid": 0, "unchanged": 0, "worsened": 0})
        per_cat[cat]["total"] += 1
        orig = e.get("original_snippet", "")
        mut = e.get("mutated_snippet", "")

        # Determine the neighborhood and after-fix snippet
        after = None
        if mut and mut in corrupted:
            # try to find replacement around mutated snippet in revised
            idx = corrupted.find(mut)
            window = corrupted[max(0, idx-200): idx+200]
            # pick a few words from orig as hint
            hint = orig[:50]
            if hint and hint in revised:
                after = hint
        if after is None:
            # fallback: search original snippet directly
            after = orig if orig in revised else None

        status = "unchanged"
        notes = ""
        if cat == "typo":
            if orig and orig in revised:
                status = "fixed_to_baseline"
            elif mut and mut in revised:
                status = "unchanged"
            else:
                status = "fixed_to_valid"
        elif cat == "link":
            # simple: link markdown well-formed
            wellformed = re.search(r"\[[^\]]+\]\([^\s)]+\)", revised) is not None
            status = "fixed_to_valid" if wellformed else "unchanged"
        elif cat == "duplicate":
            dup_before = corrupted.count(mut)
            dup_after = revised.count(mut)
            status = "fixed_to_valid" if dup_after < dup_before else "unchanged"
        elif cat == "markdown_structure":
            issues_before = _count_markdown_issues(corrupted)
            issues_after = _count_markdown_issues(revised)
            status = "fixed_to_valid" if issues_after < issues_before else "unchanged"
        elif cat in ("bio_term", "function"):
            if orig and orig in revised:
                status = "fixed_to_baseline"
            elif mut and mut in revised:
                status = "unchanged"
            else:
                status = "fixed_to_valid"
        else:
            status = "unchanged"

        mark(cat, status)
        per_error.append({
            "id": e.get("id"),
            "category": cat,
            "status": status,
            "before": mut,
            "after_contains_original": bool(orig and orig in revised),
            "notes": notes,
        })

    # global metrics
    issues_before = _count_markdown_issues(corrupted)
    issues_after = _count_markdown_issues(revised)
    global_metrics = {
        "markdown_validity_delta": issues_before - issues_after,
    }
    return {
        "per_error": per_error,
        "per_category": per_cat,
        "global": global_metrics,
    }


