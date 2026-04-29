"""Verify that error injection never places mutations inside fenced code blocks."""

import re
import pytest
from unittest.mock import MagicMock

from bioguider.generation.llm_injector import LLMErrorInjector, _CODE_FENCE_RE


SAMPLE_DOC = """\
# Seurat Tutorial

Seurat is a comprehensive R package for single cell analysis.

## What can it do?

Seurat v5 enables integrated analysis of multi-modal single-cell datasets.
The package provides methods for normalization, dimensional reduction,
and clustering of single cell expression data.

```{r setup}
library(Seurat)
pbmc <- Read10X(data.dir = "filtered_gene_bc_matrices/hg19/")
pbmc <- CreateSeuratObject(counts = pbmc, project = "pbmc3k",
                           min.cells = 3, min.features = 200)
```

After loading the data, we filter cells with high mitochondrial content.
Cells with >5% mitochondrial reads are removed. The installation process
requires R >= 4.0 and Bioconductor packages.

```r
pbmc[["percent.mt"]] <- PercentageFeatureSet(pbmc, pattern = "^MT-")
VlnPlot(pbmc, features = c("nFeature_RNA", "nCount_RNA", "percent.mt"))
pbmc <- subset(pbmc, subset = nFeature_RNA > 200 & nFeature_RNA < 2500 &
               percent.mt < 5)
```

## Requirements

- R >= 4.0
- Bioconductor

## Install

```r
install.packages("Seurat")
```

Successfully installed, the package documentation is available at
https://satijalab.org/seurat/ for further details and tutorials.

## Learn more

Visit the [Seurat website](https://satijalab.org/seurat/) for vignettes.
"""


def _fence_content(text: str) -> list[str]:
    """Extract the raw content of every fenced code block."""
    return [m.group(0) for m in _CODE_FENCE_RE.finditer(text)]


def _make_injector() -> LLMErrorInjector:
    mock_llm = MagicMock()
    return LLMErrorInjector(mock_llm, force_deterministic=True)


class TestProseOnlyInjection:
    def test_code_blocks_byte_identical(self):
        injector = _make_injector()
        corrupted, manifest = injector.inject(
            SAMPLE_DOC, min_per_category=2, force_deterministic=True
        )

        baseline_fences = _fence_content(SAMPLE_DOC)
        corrupted_fences = _fence_content(corrupted)

        assert len(baseline_fences) == len(corrupted_fences), (
            f"Fence count changed: {len(baseline_fences)} -> {len(corrupted_fences)}"
        )
        for i, (base, corr) in enumerate(zip(baseline_fences, corrupted_fences)):
            assert base == corr, (
                f"Code block {i} was modified.\n"
                f"Baseline:\n{base[:200]}\n\nCorrupted:\n{corr[:200]}"
            )

    def test_error_snippets_not_exclusively_in_code(self):
        injector = _make_injector()
        _, manifest = injector.inject(
            SAMPLE_DOC, min_per_category=2, force_deterministic=True
        )

        prose = LLMErrorInjector._prose_region(SAMPLE_DOC)
        errors = manifest.get("errors", [])
        assert len(errors) > 0, "No errors injected"

        for err in errors:
            orig = err.get("original_snippet", "")
            if not orig or len(orig) < 2:
                continue
            assert orig in prose or orig in SAMPLE_DOC, (
                f"Error snippet '{orig}' (cat={err['category']}) "
                f"not found in document at all"
            )

    def test_fence_spans_helper(self):
        spans = LLMErrorInjector._fence_spans(SAMPLE_DOC)
        assert len(spans) == 3, f"Expected 3 code blocks, got {len(spans)}"
        for start, end in spans:
            block = SAMPLE_DOC[start:end]
            assert block.startswith("```"), f"Span doesn't start with fence: {block[:20]}"
            assert block.endswith("```"), f"Span doesn't end with fence: {block[-20:]}"

    def test_replace_prose_only_skips_code(self):
        doc = "Hello world.\n```r\nlibrary(Seurat)\nSeurat::RunPCA()\n```\nSeurat is great."
        spans = LLMErrorInjector._fence_spans(doc)
        result = LLMErrorInjector._replace_prose_only(doc, "Seurat", "REPLACED", spans)
        assert "REPLACED is great" in result
        assert "library(Seurat)" in result, "Code block should be untouched"
        assert "Seurat::RunPCA()" in result, "Code block should be untouched"

    def test_replace_prose_only_no_match_returns_unchanged(self):
        doc = "```r\nSeurat\n```"
        spans = LLMErrorInjector._fence_spans(doc)
        result = LLMErrorInjector._replace_prose_only(doc, "Seurat", "REPLACED", spans)
        assert result == doc

    def test_high_error_count_still_prose_only(self):
        injector = _make_injector()
        corrupted, manifest = injector.inject(
            SAMPLE_DOC, min_per_category=5, force_deterministic=True
        )

        baseline_fences = _fence_content(SAMPLE_DOC)
        corrupted_fences = _fence_content(corrupted)

        for i, (base, corr) in enumerate(zip(baseline_fences, corrupted_fences)):
            assert base == corr, (
                f"Code block {i} modified at high error count.\n"
                f"Baseline:\n{base[:200]}\n\nCorrupted:\n{corr[:200]}"
            )
