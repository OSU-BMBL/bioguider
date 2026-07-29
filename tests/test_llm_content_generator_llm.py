"""Regression tests pinning that ``LLMContentGenerator`` always uses the
LLM it was constructed with — never a hard-coded default model.

Background: ``_generate_continuation`` used to call
``get_openai()`` (a default Azure/OpenAI model built from env vars)
instead of ``self.llm``.  When a model's first generation pass truncated,
the continuation was silently written by a *different* model than the one
under test, contaminating per-model benchmark results.  The debug log
likewise reported the env ``OPENAI_MODEL`` rather than the model in use.

These tests run without real LLM calls by patching ``CommonConversation``
at the module level.
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from bioguider.generation.llm_content_generator import LLMContentGenerator


class _SentinelLLM:
    """A stand-in LLM that is *not* what ``get_openai()`` would return.

    Carries the attributes the debug logger reads so we can assert they
    are sourced from the instance, not the environment.
    """

    model_name = "kimi-k2.5"
    deployment_name = "kimi-deploy"
    max_tokens = 4096


def _patch_conv():
    """Patch CommonConversation in the generator module; return the mock
    class whose instance's ``generate`` yields a (content, usage) tuple."""
    p = patch("bioguider.generation.llm_content_generator.CommonConversation")
    MockConv = p.start()
    MockConv.return_value.generate.return_value = (
        "continued body text", {"total_tokens": 7, "prompt_tokens": 3, "completion_tokens": 4},
    )
    return p, MockConv


# ---------------------------------------------------------------------------
# _generate_continuation must use self.llm
# ---------------------------------------------------------------------------

class TestContinuationUsesPassedLLM:
    def test_continuation_builds_conversation_with_self_llm(self):
        llm = _SentinelLLM()
        gen = LLMContentGenerator(llm)
        p, MockConv = _patch_conv()
        try:
            content, usage = gen._generate_continuation(
                target_file="guide.md",
                evaluation_report={"total_suggestions": 2},
                context="repo context",
                existing_content="some partial document text",
            )
        finally:
            p.stop()
        # The conversation must be constructed from the generator's own LLM,
        # not a hard-coded default.
        MockConv.assert_called_once_with(llm)
        assert content == "continued body text"
        assert usage["total_tokens"] == 7

    def test_continuation_source_does_not_reference_get_openai(self):
        """The hard-coded default-model factory must not be referenced at all.

        Source-level check (rather than patching ``get_openai``, which would
        force importing the langchain-backed ``agent_utils``) so the
        regression is pinned regardless of environment.
        """
        import inspect

        src = inspect.getsource(LLMContentGenerator._generate_continuation)
        assert "get_openai" not in src
        assert "self.llm" in src


# ---------------------------------------------------------------------------
# Debug log must report the in-use model, not the env default
# ---------------------------------------------------------------------------

class TestDebugModelNameReflectsLLM:
    def _read_debug(self, tmp_path, target_file):
        safe = target_file.replace("/", "_").replace(".", "_")
        debug_file = tmp_path / "outputs" / "debug_generation" / f"{safe}_debug.json"
        return json.loads(debug_file.read_text(encoding="utf-8"))

    def test_debug_uses_llm_attrs_over_env(self, tmp_path, monkeypatch):
        # Env points at a DIFFERENT model — the debug log must ignore it.
        monkeypatch.setenv("OPENAI_MODEL", "azure-default-gpt")
        monkeypatch.setenv("OPENAI_DEPLOYMENT_NAME", "azure-default-deploy")
        monkeypatch.chdir(tmp_path)

        gen = LLMContentGenerator(_SentinelLLM())
        p, _MockConv = _patch_conv()
        try:
            gen.generate_full_document(
                target_file="guide.md",
                evaluation_report={"total_suggestions": 1, "suggestions": [{"x": 1}]},
                context="ctx",
                original_content="# short original\n",
            )
        finally:
            p.stop()

        debug = self._read_debug(tmp_path, "guide.md")
        assert debug["llm_settings"]["model_name"] == "kimi-k2.5"
        assert debug["llm_settings"]["azure_deployment"] == "kimi-deploy"
        assert debug["llm_settings"]["max_tokens"] == 4096

    def test_debug_falls_back_to_env_when_llm_lacks_attrs(self, tmp_path, monkeypatch):
        monkeypatch.setenv("OPENAI_MODEL", "env-fallback-model")
        monkeypatch.setenv("OPENAI_DEPLOYMENT_NAME", "env-fallback-deploy")
        monkeypatch.chdir(tmp_path)

        bare_llm = MagicMock(spec=[])  # no model_name / deployment_name / max_tokens
        gen = LLMContentGenerator(bare_llm)
        p, _MockConv = _patch_conv()
        try:
            gen.generate_full_document(
                target_file="guide.md",
                evaluation_report={"total_suggestions": 1},
                context="ctx",
                original_content="# short original\n",
            )
        finally:
            p.stop()

        debug = self._read_debug(tmp_path, "guide.md")
        assert debug["llm_settings"]["model_name"] == "env-fallback-model"
        assert debug["llm_settings"]["azure_deployment"] == "env-fallback-deploy"
