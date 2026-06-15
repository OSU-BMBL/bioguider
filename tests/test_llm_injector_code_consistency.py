"""
Unit tests for code-consistency error injection:
  code_func_name, code_func_args, code_comment_conflict

These three categories are CODE_INTERNAL_CATEGORIES — their mutation lives
inside ``` fences, which is unfixable by a documentation evaluator. They are
therefore scoped OUT of injection entirely (regardless of file type) and only
recorded in the manifest's "skipped" list with reason
"scoped_out_code_internal". The tests below assert that contract.

All tests use _deterministic_inject (no LLM) or _inject_code_consistency
directly so they run fast without network calls.
"""
import pytest
from bioguider.generation.llm_injector import (
    LLMErrorInjector,
    _COMMENT_CONFLICT_MAP,
    CODE_INTERNAL_CATEGORIES,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

DOC_WITH_CODE = """\
# Seurat Tutorial

We load and cluster cells.

```r
# normalize the data
NormalizeData(pbmc, normalization.method = "LogNormalize")
pbmc <- FindNeighbors(pbmc, dims = 1:10)
pbmc <- FindClusters(pbmc, resolution = 0.5)
```

More prose here.
"""

DOC_WITHOUT_CODE = """\
# Seurat Tutorial

We load and cluster cells. No code blocks here.
Just plain prose describing the analysis.
"""


def _make_injector():
    return LLMErrorInjector(llm=None, force_deterministic=True)


def _categories(manifest):
    return {e["category"] for e in manifest.get("errors", [])}


def _skipped_categories(manifest):
    return {s["category"] for s in manifest.get("skipped", [])}


# ---------------------------------------------------------------------------
# _inject_code_consistency — direct tests
# ---------------------------------------------------------------------------

class TestInjectCodeConsistencyDirect:
    def _call(self, text):
        inj = _make_injector()
        errors = []
        data = {}
        result, errors = inj._inject_code_consistency(text, errors, data)
        return result, errors, data

    def test_code_func_name_not_injected(self):
        # Removed from injection: corrupting executable R code is out of scope
        # for a documentation evaluator. Category lives in UNSCORABLE_CATEGORIES.
        result, errors, data = self._call(DOC_WITH_CODE)
        cats = {e["category"] for e in errors}
        assert "code_func_name" not in cats, f"code_func_name should not be injected: {cats}"

    def test_code_func_name_mutation_is_in_code_block(self):
        result, errors, data = self._call(DOC_WITH_CODE)
        fn_errors = [e for e in errors if e["category"] == "code_func_name"]
        if not fn_errors:
            pytest.skip("no code_func_name error injected")
        orig = fn_errors[0]["original_snippet"]
        # The original snippet must appear in the original doc's code block
        assert orig in DOC_WITH_CODE

    def test_code_func_name_not_in_prose(self):
        """Mutation must only appear inside a code block, not in prose."""
        result, errors, data = self._call(DOC_WITH_CODE)
        fn_errors = [e for e in errors if e["category"] == "code_func_name"]
        if not fn_errors:
            pytest.skip("no code_func_name error injected")
        mutated = fn_errors[0]["mutated_snippet"]
        prose = result.split("```")[0]  # text before first fence
        assert mutated not in prose

    def test_code_func_args_not_injected(self):
        # Removed from injection: corrupting executable R code is out of scope.
        result, errors, data = self._call(DOC_WITH_CODE)
        cats = {e["category"] for e in errors}
        assert "code_func_args" not in cats, f"code_func_args should not be injected: {cats}"

    def test_code_func_args_snippet_format(self):
        result, errors, data = self._call(DOC_WITH_CODE)
        arg_errors = [e for e in errors if e["category"] == "code_func_args"]
        if not arg_errors:
            pytest.skip("no code_func_args error injected")
        # original_snippet should be the bare arg name (no trailing spaces after strip)
        orig = arg_errors[0]["original_snippet"]
        assert "=" in orig

    def test_code_comment_conflict_not_injected(self):
        # Scoped out: in-fence comment mutation is unfixable by a doc evaluator.
        result, errors, data = self._call(DOC_WITH_CODE)
        cats = {e["category"] for e in errors}
        assert "code_comment_conflict" not in cats, f"should not be injected: {cats}"

    def test_code_internal_categories_recorded_skipped(self):
        result, errors, data = self._call(DOC_WITH_CODE)
        skipped = {s["category"]: s["reason"] for s in data.get("skipped", [])}
        for cat in ("code_func_name", "code_func_args", "code_comment_conflict"):
            assert skipped.get(cat) == "scoped_out_code_internal", (
                f"{cat} should be skipped as scoped_out_code_internal: {skipped}"
            )

    def test_no_code_block_records_comment_conflict_skipped(self):
        # All three code-internal categories are scoped out, so errors is empty
        # and they appear only in skipped.
        _, errors, data = self._call(DOC_WITHOUT_CODE)
        skipped = _skipped_categories(data)
        assert "code_comment_conflict" in skipped
        assert errors == []

    def test_code_blocks_preserved_after_injection(self):
        result, _, _ = self._call(DOC_WITH_CODE)
        assert _make_injector()._check_code_blocks_preserved(DOC_WITH_CODE, result)


# ---------------------------------------------------------------------------
# Full deterministic path (inject → supplement)
# ---------------------------------------------------------------------------

class TestDeterministicPathCodeConsistency:
    def test_full_inject_does_not_produce_code_internal(self):
        inj = _make_injector()
        corrupted, manifest = inj.inject(DOC_WITH_CODE, min_per_category=1)
        cats = _categories(manifest)
        # All code-internal categories are scoped out of injection.
        leaked = cats & CODE_INTERNAL_CATEGORIES
        assert not leaked, f"code-internal categories leaked into manifest: {leaked}"

    def test_full_inject_preserves_code_blocks(self):
        inj = _make_injector()
        corrupted, manifest = inj.inject(DOC_WITH_CODE, min_per_category=1)
        assert inj._check_code_blocks_preserved(DOC_WITH_CODE, corrupted)

    def test_no_code_doc_records_comment_conflict_skipped(self):
        inj = _make_injector()
        _, manifest = inj.inject(DOC_WITHOUT_CODE, min_per_category=1)
        skipped = _skipped_categories(manifest)
        # code_func_name / code_func_args no longer injected, so not in skipped
        assert "code_comment_conflict" in skipped


# ---------------------------------------------------------------------------
# _replace_in_fence — unit tests
# ---------------------------------------------------------------------------

class TestReplaceInFence:
    def _spans(self, text):
        return LLMErrorInjector._fence_spans(text)

    def test_replaces_inside_fence(self):
        text = "prose\n```r\nfoo(\n```\nmore prose"
        spans = self._spans(text)
        result = LLMErrorInjector._replace_in_fence(text, "foo(", "bar(", spans)
        assert "bar(" in result
        assert "foo(" not in result

    def test_does_not_replace_in_prose(self):
        text = "foo(\n```r\nfoo(\n```"
        spans = self._spans(text)
        result = LLMErrorInjector._replace_in_fence(text, "foo(", "bar(", spans)
        # Only the second foo( (inside fence) should be replaced
        assert result.count("foo(") == 1
        assert result.startswith("foo(")  # prose occurrence untouched

    def test_returns_unchanged_when_not_in_fence(self):
        text = "foo() in prose only, no fences"
        spans = self._spans(text)
        result = LLMErrorInjector._replace_in_fence(text, "foo(", "bar(", spans)
        assert result == text

    def test_returns_unchanged_when_old_not_found(self):
        text = "```r\nbar(\n```"
        spans = self._spans(text)
        result = LLMErrorInjector._replace_in_fence(text, "xyz(", "abc(", spans)
        assert result == text


# ---------------------------------------------------------------------------
# _COMMENT_CONFLICT_MAP — semantic flip tests
# ---------------------------------------------------------------------------

class TestCommentConflictMap:
    def _apply(self, text):
        for pat, replacement in _COMMENT_CONFLICT_MAP:
            if pat.search(text):
                return pat.sub(replacement, text, count=1)
        return None

    def test_normalize_maps_to_scale(self):
        result = self._apply("normalize the data")
        assert result == "scale the data"

    def test_cluster_maps_to_normalize(self):
        result = self._apply("cluster the cells")
        assert result == "normalize the cells"

    def test_filter_maps_to_cluster(self):
        result = self._apply("filter low-quality cells")
        assert result == "cluster low-quality cells"

    def test_load_maps_to_save(self):
        result = self._apply("load the dataset")
        assert result == "save the dataset"

    def test_no_match_returns_none(self):
        result = self._apply("perform analysis")
        assert result is None


# ---------------------------------------------------------------------------
# File-type-aware injection tests
# ---------------------------------------------------------------------------

class TestFileTypeAwareInjection:
    """All code-internal categories (code_func_name, code_func_args,
    code_comment_conflict) are scoped out of injection for EVERY file type and
    recorded in "skipped" with reason "scoped_out_code_internal"."""

    def _call(self, text, file_type=""):
        inj = _make_injector()
        errors = []
        data = {}
        result, errors = inj._inject_code_consistency(text, errors, data, file_type=file_type)
        return result, errors, data

    def _assert_scoped_out(self, file_type):
        result, errors, data = self._call(DOC_WITH_CODE, file_type=file_type)
        assert errors == [], f"Expected no errors for {file_type!r}, got: {errors}"
        skipped = {s["category"]: s["reason"] for s in data.get("skipped", [])}
        for cat in ("code_func_name", "code_func_args", "code_comment_conflict"):
            assert skipped.get(cat) == "scoped_out_code_internal", (
                f"{cat} for {file_type!r}: {skipped}"
            )

    def test_md_scopes_out_code_internal(self):
        self._assert_scoped_out(".md")

    def test_rst_scopes_out_code_internal(self):
        self._assert_scoped_out(".rst")

    def test_rmd_scopes_out_code_internal(self):
        self._assert_scoped_out(".rmd")

    def test_ipynb_scopes_out_code_internal(self):
        self._assert_scoped_out(".ipynb")

    def test_default_scopes_out_code_internal(self):
        self._assert_scoped_out("")
