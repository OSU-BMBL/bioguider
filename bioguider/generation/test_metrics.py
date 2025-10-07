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
    # aggregate counters
    totals = {"total_errors": 0, "fixed_to_baseline": 0, "fixed_to_valid": 0, "unchanged": 0, "worsened": 0}

    def mark(cat: str, key: str):
        per_cat.setdefault(cat, {"total": 0, "fixed_to_baseline": 0, "fixed_to_valid": 0, "unchanged": 0, "worsened": 0})
        per_cat[cat][key] += 1

    # Precompute some structural counts
    def count_malformed_bullets(text: str) -> int:
        return len(re.findall(r"^[-*]\S", text, flags=re.M))

    def count_bad_image_spacing(text: str) -> int:
        return len(re.findall(r"!\[[^\]]*\]\s+\(", text))

    def table_variance(text: str) -> int:
        rows = [ln for ln in text.splitlines() if '|' in ln]
        groups: List[List[str]] = []
        cur: List[str] = []
        for ln in rows:
            if '|' in ln:
                cur.append(ln)
            else:
                if len(cur) >= 2:
                    groups.append(cur)
                cur = []
        if len(cur) >= 2:
            groups.append(cur)
        vari = 0
        for g in groups:
            counts = [ln.count('|') for ln in g]
            vari += (max(counts) - min(counts))
        return vari

    malformed_bullets_before = count_malformed_bullets(corrupted)
    malformed_bullets_after = count_malformed_bullets(revised)
    bad_img_before = count_bad_image_spacing(corrupted)
    bad_img_after = count_bad_image_spacing(revised)
    table_var_before = table_variance(corrupted)
    table_var_after = table_variance(revised)

    canonical_titles = {
        "## What is it?",
        "## What can it do?",
        "## Requirements",
        "## Install",
        "## Quick example",
        "## Learn more",
        "## License & Contact",
    }

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
        elif cat == "list_structure":
            status = "fixed_to_valid" if malformed_bullets_after < malformed_bullets_before else "unchanged"
        elif cat == "image_syntax":
            status = "fixed_to_valid" if bad_img_after < bad_img_before else "unchanged"
        elif cat == "section_title":
            # valid if mutated title removed and any canonical title present
            if mut and mut not in revised and any(t in revised for t in canonical_titles):
                status = "fixed_to_valid"
            else:
                status = "unchanged"
        elif cat == "inline_code":
            # check that the raw content regained backticks somewhere
            raw = mut.strip('`') if mut else ""
            rewrapped = f"`{raw}`" if raw else ""
            if raw and rewrapped and rewrapped in revised and mut not in revised:
                status = "fixed_to_valid"
            else:
                status = "unchanged"
        elif cat == "emphasis":
            status = "fixed_to_valid" if mut and mut not in revised else "unchanged"
        elif cat == "table_alignment":
            status = "fixed_to_valid" if table_var_after < table_var_before else "unchanged"
        elif cat == "code_lang_tag":
            status = "fixed_to_valid" if mut and mut not in revised else "unchanged"
        # Biology-specific and CLI/CONFIG categories: treat as fixed if mutated snippet removed
        elif cat in {
            "gene_symbol_case","species_swap","ref_genome_mismatch","modality_confusion","normalization_error",
            "umi_vs_read","batch_effect","qc_threshold","file_format","strandedness","coordinates","units_scale",
            "sample_type","contamination","param_name","default_value","path_hint"
        }:
            status = "fixed_to_valid" if mut and mut not in revised else "unchanged"
        else:
            status = "unchanged"

        mark(cat, status)
        totals["total_errors"] += 1
        totals[status] += 1
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
    success = totals["fixed_to_baseline"] + totals["fixed_to_valid"]
    success_rate = (success / totals["total_errors"] * 100.0) if totals["total_errors"] else 0.0
    summary = {
        "totals": totals,
        "success_rate": round(success_rate, 2),
    }
    return {
        "per_error": per_error,
        "per_category": per_cat,
        "global": global_metrics,
        "summary": summary,
    }


