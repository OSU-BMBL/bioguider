"""D6 — ``force_deterministic`` knob on ``LLMErrorInjector``.

Reproducible cross-model injection: when ``force_deterministic=True`` we skip
the LLM entirely, so every model in the horizontal benchmark matrix sees
byte-identical corrupted files.
"""

from unittest.mock import MagicMock

from bioguider.generation.llm_injector import LLMErrorInjector


FIXTURE = """
# Title

We used Seurat v5 at resolution of 0.5. Data from series GSE123456.

```r
library(Seurat_5.0.1)
FindClusters(obj, resolution = 0.5)
wilcox.test(x, y)
subset(obj, CD8 > 0)
```
"""


class TestInstanceLevelFlag:
    def test_default_is_false(self):
        injector = LLMErrorInjector(MagicMock())
        assert injector.force_deterministic is False

    def test_flag_passed_in_ctor(self):
        injector = LLMErrorInjector(MagicMock(), force_deterministic=True)
        assert injector.force_deterministic is True


class TestLLMPathSkippedWhenForced:
    def test_llm_never_invoked_when_instance_flag_true(self):
        llm = MagicMock()
        injector = LLMErrorInjector(llm, force_deterministic=True)
        # Ensure CommonConversation.generate would blow up if reached — force
        # deterministic path must NOT touch the LLM at all.
        injector._parse_json_output = lambda *_a, **_k: pytest_fail("LLM path entered")
        _, manifest = injector.inject(FIXTURE, min_per_category=1)
        # LLM never called because we short-circuited before CommonConversation.
        assert llm.mock_calls == []
        assert isinstance(manifest["errors"], list)

    def test_per_call_override_true_skips_llm(self):
        llm = MagicMock()
        injector = LLMErrorInjector(llm, force_deterministic=False)
        injector._parse_json_output = lambda *_a, **_k: pytest_fail("LLM path entered")
        _, manifest = injector.inject(FIXTURE, min_per_category=1, force_deterministic=True)
        assert llm.mock_calls == []
        assert isinstance(manifest["errors"], list)


class TestByteIdenticalAcrossCalls:
    """Phase 1 depends on this: two runs produce identical corrupted text."""

    def test_two_runs_produce_same_corrupted_output(self):
        injector_a = LLMErrorInjector(MagicMock(), force_deterministic=True)
        injector_b = LLMErrorInjector(MagicMock(), force_deterministic=True)
        corrupted_a, manifest_a = injector_a.inject(FIXTURE, min_per_category=1)
        corrupted_b, manifest_b = injector_b.inject(FIXTURE, min_per_category=1)
        assert corrupted_a == corrupted_b
        # Error category lists should also match order-for-order.
        cats_a = [e["category"] for e in manifest_a["errors"]]
        cats_b = [e["category"] for e in manifest_b["errors"]]
        assert cats_a == cats_b


class TestStillHitsAnchoredCategories:
    """Deterministic path must still fire the D1 prose-code categories."""

    def test_prose_code_categories_fire(self):
        injector = LLMErrorInjector(MagicMock(), force_deterministic=True)
        _, manifest = injector.inject(FIXTURE, min_per_category=1)
        cats = {e["category"] for e in manifest["errors"]}
        fired = cats & {
            "prose_code_pkg_version",
            "prose_code_stat_test",
            "prose_code_marker",
            "prose_code_param",
            "accession_id_prefix",
        }
        assert len(fired) >= 2, f"Expected anchored categories to fire; got {cats}"


def pytest_fail(msg):
    import pytest as _p

    _p.fail(msg)
