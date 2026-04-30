"""
Unit tests for check_protected_regions() and count_collateral_damage()
in bioguider/generation/benchmark_metrics.py.
"""

from bioguider.generation.benchmark_metrics import (
    check_protected_regions,
    count_collateral_damage,
)


SAMPLE_DOC = """\
---
title: "Test Vignette"
output: html_document
---

# Introduction

This tutorial uses Seurat for single-cell analysis.

```{r setup}
library(Seurat)
```

## Data Loading

Load the PBMC dataset with 2700 cells.

```{r load}
pbmc <- Read10X("data/")
```

## Analysis

We use PCA for dimensionality reduction.
"""


# ---------------------------------------------------------------------------
# check_protected_regions
# ---------------------------------------------------------------------------


class TestCheckProtectedRegions:
    def test_identical_docs(self):
        result = check_protected_regions(SAMPLE_DOC, SAMPLE_DOC)
        assert result["code_fence_violations"] == 0
        assert result["yaml_violations"] == 0
        assert result["section_violations"] == 0

    def test_code_fence_content_changed(self):
        revised = SAMPLE_DOC.replace("library(Seurat)", "library(Seurat2)")
        result = check_protected_regions(SAMPLE_DOC, revised)
        assert result["code_fence_violations"] >= 1

    def test_code_fence_header_changed(self):
        # Change ```{r setup} to ```{r} — header altered, content unchanged
        revised = SAMPLE_DOC.replace("```{r setup}", "```{r}")
        result = check_protected_regions(SAMPLE_DOC, revised)
        assert result["code_fence_violations"] >= 1

    def test_code_fence_added(self):
        extra_fence = "\n```{r extra}\nx <- 1\n```\n"
        revised = SAMPLE_DOC + extra_fence
        result = check_protected_regions(SAMPLE_DOC, revised)
        assert result["code_fence_violations"] >= 1

    def test_yaml_changed(self):
        revised = SAMPLE_DOC.replace(
            'output: html_document', 'output: pdf_document'
        )
        result = check_protected_regions(SAMPLE_DOC, revised)
        assert result["yaml_violations"] == 1

    def test_yaml_missing_both(self):
        doc_no_yaml = "# Intro\n\nSome prose here.\n"
        result = check_protected_regions(doc_no_yaml, doc_no_yaml)
        assert result["yaml_violations"] == 0

    def test_section_header_changed(self):
        revised = SAMPLE_DOC.replace("## Data Loading", "## Data Import")
        result = check_protected_regions(SAMPLE_DOC, revised)
        assert result["section_violations"] >= 1

    def test_section_header_deleted(self):
        revised = SAMPLE_DOC.replace("## Analysis\n\n", "")
        result = check_protected_regions(SAMPLE_DOC, revised)
        assert result["section_violations"] >= 1

    def test_prose_only_change(self):
        # Only change prose — no protected region touched
        revised = SAMPLE_DOC.replace(
            "Load the PBMC dataset with 2700 cells.",
            "Load the PBMC dataset with 3000 cells.",
        )
        result = check_protected_regions(SAMPLE_DOC, revised)
        assert result["code_fence_violations"] == 0
        assert result["yaml_violations"] == 0
        assert result["section_violations"] == 0


# ---------------------------------------------------------------------------
# count_collateral_damage
# ---------------------------------------------------------------------------


class TestCountCollateralDamage:
    def test_no_changes(self):
        result = count_collateral_damage(SAMPLE_DOC, SAMPLE_DOC, [])
        assert result == []

    def test_error_fix_not_counted(self):
        # The model fixed "Suerat" → "Seurat" — matches mutated_snippet in baseline.
        errors = [
            {
                "original_snippet": "Seurat",
                "mutated_snippet": "Suerat",
                "category": "typo",
            }
        ]
        corrupted = SAMPLE_DOC.replace("Seurat for single-cell", "Suerat for single-cell")
        # revised restores the correct text
        revised = SAMPLE_DOC
        result = count_collateral_damage(corrupted, revised, errors)
        assert result == []

    def test_collateral_change_detected(self):
        # Change prose that has nothing to do with any injected error
        revised = SAMPLE_DOC.replace(
            "We use PCA for dimensionality reduction.",
            "We use UMAP for dimensionality reduction.",
        )
        errors = [
            {
                "original_snippet": "Seurat",
                "mutated_snippet": "Suerat",
                "category": "typo",
            }
        ]
        result = count_collateral_damage(SAMPLE_DOC, revised, errors)
        assert len(result) == 1
        assert "PCA" in result[0]["original"]
        assert "UMAP" in result[0]["changed"]

    def test_protected_region_excluded(self):
        # Change only inside a code fence — should NOT be counted as collateral
        revised = SAMPLE_DOC.replace("library(Seurat)", "library(Seurat2)")
        result = count_collateral_damage(SAMPLE_DOC, revised, [])
        assert result == []

    def test_yaml_change_excluded(self):
        # Change only inside YAML frontmatter — should NOT be counted as collateral
        revised = SAMPLE_DOC.replace(
            'output: html_document', 'output: pdf_document'
        )
        result = count_collateral_damage(SAMPLE_DOC, revised, [])
        assert result == []

    def test_multiple_collateral(self):
        # Two separate prose changes outside any error snippet
        revised = SAMPLE_DOC.replace(
            "This tutorial uses Seurat for single-cell analysis.",
            "This guide uses Seurat for single-cell analysis.",
        ).replace(
            "We use PCA for dimensionality reduction.",
            "We use tSNE for dimensionality reduction.",
        )
        errors: list = []
        result = count_collateral_damage(SAMPLE_DOC, revised, errors)
        assert len(result) == 2
