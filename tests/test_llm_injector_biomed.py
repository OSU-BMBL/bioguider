"""Tests for anchor-required prose-code consistency + accession-id injection.

These replace the earlier ``biomed_app`` tests. The five old types
(reproducibility_drift, analysis_hyperparam, stat_test_misnaming,
annotation_id_space, celltype_marker) are dropped or rehomed:

- reproducibility_drift (seed bump)  — dropped entirely (no ground truth).
- analysis_hyperparam (raw number)    — rehomed as ``prose_code_param``.
- stat_test_misnaming                 — rehomed as ``prose_code_stat_test``.
- annotation_id_space                 — renamed to ``accession_id_prefix``
                                        and now requires a context-word anchor
                                        (``series`` / ``samples`` / ...).
- celltype_marker                     — rehomed as ``prose_code_marker``.
- NEW: ``prose_code_pkg_version``     — prose pkg version disagrees with a
                                        version pinned in a code fence.

All four ``prose_code_*`` categories demand a code-block anchor; if absent the
injector records ``{"category": ..., "reason": "no_anchor"}`` in
``manifest["skipped"]``.
"""

from unittest.mock import MagicMock

from bioguider.generation.llm_injector import LLMErrorInjector


FIXTURE_WITH_CODE_ANCHORS = """
# Analysis vignette

We used Seurat v5 for clustering at resolution of 0.5, called CD8+ T cells,
and ran a Wilcoxon rank-sum test per cluster. Data come from series GSE123456.

```r
library(Seurat_5.0.1)
obj <- FindClusters(obj, resolution = 0.5)
markers <- FindMarkers(obj, test.use = "wilcox")
cd8 <- subset(obj, CD8 > 0)
```
"""

FIXTURE_NO_CODE_ANCHORS = """
# Overview

We used Seurat v4 for clustering, called CD8+ T cells, and ran a Wilcoxon
test per cluster. Data come from GSE123456 (no context word here).
"""


def _make_injector():
    """Build an injector whose LLM never returns edits so _supplement_errors runs."""
    llm = MagicMock()
    injector = LLMErrorInjector(llm)
    injector._parse_json_output = lambda output, fallback: {
        "corrupted_markdown": fallback,
        "errors": [],
    }
    return injector


class TestProseCodeConsistencySupplement:
    """AC: ``_supplement_errors`` injects each category when its anchor exists."""

    def test_prose_code_pkg_version_injected(self):
        injector = _make_injector()
        _, manifest = injector.inject(FIXTURE_WITH_CODE_ANCHORS, min_per_category=1)
        cats = {e["category"] for e in manifest["errors"]}
        assert "prose_code_pkg_version" in cats, f"Got: {cats}"

    def test_prose_code_stat_test_injected(self):
        injector = _make_injector()
        _, manifest = injector.inject(FIXTURE_WITH_CODE_ANCHORS, min_per_category=1)
        cats = {e["category"] for e in manifest["errors"]}
        assert "prose_code_stat_test" in cats, f"Got: {cats}"

    def test_prose_code_marker_injected(self):
        injector = _make_injector()
        _, manifest = injector.inject(FIXTURE_WITH_CODE_ANCHORS, min_per_category=1)
        cats = {e["category"] for e in manifest["errors"]}
        assert "prose_code_marker" in cats, f"Got: {cats}"

    def test_prose_code_param_injected(self):
        injector = _make_injector()
        _, manifest = injector.inject(FIXTURE_WITH_CODE_ANCHORS, min_per_category=1)
        cats = {e["category"] for e in manifest["errors"]}
        assert "prose_code_param" in cats, f"Got: {cats}"

    def test_accession_id_prefix_injected_with_context_word(self):
        injector = _make_injector()
        _, manifest = injector.inject(FIXTURE_WITH_CODE_ANCHORS, min_per_category=1)
        cats = {e["category"] for e in manifest["errors"]}
        assert "accession_id_prefix" in cats, f"Got: {cats}"


class TestSkipWhenAnchorMissing:
    """AC: when the required anchor is absent, category is recorded in ``skipped``."""

    def test_prose_code_pkg_version_skipped_without_code_anchor(self):
        injector = _make_injector()
        _, manifest = injector.inject(FIXTURE_NO_CODE_ANCHORS, min_per_category=1)
        cats = {e["category"] for e in manifest["errors"]}
        assert "prose_code_pkg_version" not in cats
        skipped = {s["category"]: s["reason"] for s in manifest.get("skipped", [])}
        assert skipped.get("prose_code_pkg_version") == "no_anchor"

    def test_prose_code_stat_test_skipped_without_code_anchor(self):
        injector = _make_injector()
        _, manifest = injector.inject(FIXTURE_NO_CODE_ANCHORS, min_per_category=1)
        skipped = {s["category"] for s in manifest.get("skipped", [])}
        assert "prose_code_stat_test" in skipped

    def test_prose_code_marker_skipped_without_code_anchor(self):
        injector = _make_injector()
        _, manifest = injector.inject(FIXTURE_NO_CODE_ANCHORS, min_per_category=1)
        skipped = {s["category"] for s in manifest.get("skipped", [])}
        assert "prose_code_marker" in skipped

    def test_prose_code_param_skipped_without_code_anchor(self):
        injector = _make_injector()
        _, manifest = injector.inject(FIXTURE_NO_CODE_ANCHORS, min_per_category=1)
        skipped = {s["category"] for s in manifest.get("skipped", [])}
        assert "prose_code_param" in skipped


class TestDeterministicInjectAnchored:
    """``_deterministic_inject`` honours the same anchor rules end-to-end."""

    def test_deterministic_injects_when_anchors_present(self):
        injector = LLMErrorInjector(MagicMock())
        _, data = injector._deterministic_inject(FIXTURE_WITH_CODE_ANCHORS)
        cats = {e["category"] for e in data["errors"]}
        # At least two prose-code categories should fire on this fixture.
        fired = cats & {
            "prose_code_pkg_version",
            "prose_code_stat_test",
            "prose_code_marker",
            "prose_code_param",
            "accession_id_prefix",
        }
        assert len(fired) >= 2, f"Expected >=2 anchored categories; got {cats}"

    def test_deterministic_records_skips_when_anchors_missing(self):
        injector = LLMErrorInjector(MagicMock())
        _, data = injector._deterministic_inject(FIXTURE_NO_CODE_ANCHORS)
        skipped_cats = {s["category"] for s in data.get("skipped", [])}
        expected = {
            "prose_code_pkg_version",
            "prose_code_stat_test",
            "prose_code_marker",
            "prose_code_param",
        }
        assert expected.issubset(skipped_cats), f"Missing skips: {expected - skipped_cats}"


class TestDroppedCategoriesAreGone:
    """AC: the un-benchmarkable old types must NEVER appear in a manifest."""

    DROPPED = {
        "reproducibility_drift",
        "analysis_hyperparam",
        "stat_test_misnaming",
        "annotation_id_space",
        "celltype_marker",
        "biomed_app",
    }

    def test_dropped_cats_never_in_error_categories(self):
        from bioguider.managers.config import ALL_ERROR_CATEGORIES, ERROR_CATEGORIES
        assert self.DROPPED.isdisjoint(ALL_ERROR_CATEGORIES), \
            f"Stale categories: {self.DROPPED & ALL_ERROR_CATEGORIES}"
        assert "biomed_app" not in ERROR_CATEGORIES, "biomed_app group must be removed"

    def test_new_group_present(self):
        from bioguider.managers.config import ERROR_CATEGORIES
        assert "prose_code_consistency" in ERROR_CATEGORIES
        assert set(ERROR_CATEGORIES["prose_code_consistency"]) == {
            "prose_code_pkg_version",
            "prose_code_stat_test",
            "prose_code_marker",
            "prose_code_param",
        }

    def test_accession_id_prefix_in_biology_group(self):
        from bioguider.managers.config import ERROR_CATEGORIES
        assert "accession_id_prefix" in ERROR_CATEGORIES["biology"]

    def test_dropped_cats_not_in_any_manifest(self):
        injector = _make_injector()
        _, manifest = injector.inject(FIXTURE_WITH_CODE_ANCHORS, min_per_category=2)
        cats = {e["category"] for e in manifest["errors"]}
        assert self.DROPPED.isdisjoint(cats), f"Leaked: {self.DROPPED & cats}"
