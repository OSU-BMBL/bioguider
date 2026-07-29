"""Pins the ``.fixed.<ext>`` extension plumbing in ``fix_with_model``.

Earlier the extension was hardcoded to ``.Rmd``, so the pharokka benchmark
(``docs/plotting.md``) produced ``*.fixed.Rmd`` siblings next to
``*.corrupted.md`` and ``*.pipeline_fixed.md``. The ``file_type`` parameter
now drives the suffix so benchmark output directories stay coherent.
"""
import os
import tempfile
import types

import benchmark.shared as bs


def _stub_response(text: str):
    """Build a minimal AIMessage-ish object the cleaner can chew on."""
    return types.SimpleNamespace(
        content=text,
        response_metadata={"token_usage": {"prompt_tokens": 1, "completion_tokens": 1}},
    )


def _patch_llm(monkeypatch, response_text: str):
    """Stub out the network: model constructors and the retry wrapper."""
    class _FakeChat:
        def __init__(self, *args, **kwargs):
            pass

        def invoke(self, prompt):  # pragma: no cover - exercised via _invoke_with_retry
            return _stub_response(response_text)

    monkeypatch.setattr(bs, "ChatOpenAI", _FakeChat)
    monkeypatch.setattr(bs, "_invoke_with_retry", lambda _llm, _prompt: _stub_response(response_text))


def test_default_extension_is_rmd_for_backwards_compat(monkeypatch):
    """Seurat benchmarks rely on the default — guard against accidental change."""
    _patch_llm(monkeypatch, "---\ntitle: ok\n---\n\nbody\n" * 50)
    with tempfile.TemporaryDirectory() as td:
        bs.fix_with_model(
            llm=None,
            corrupted_content="x" * 200,
            original_content="x" * 200,
            output_dir=td,
            file_basename="foo",
            error_count=10,
            prompt_name="bioguider",
            model_name="gpt-4o",
        )
        files = os.listdir(td)
    assert "foo.level_10.gpt-4o_bioguider.fixed.Rmd" in files
    assert not any(f.endswith(".fixed.md") for f in files)


def test_explicit_md_extension_matches_input(monkeypatch):
    """Pharokka benchmark needs ``.md`` so the output dir stays coherent."""
    _patch_llm(monkeypatch, "---\ntitle: ok\n---\n\nbody\n" * 50)
    with tempfile.TemporaryDirectory() as td:
        bs.fix_with_model(
            llm=None,
            corrupted_content="x" * 200,
            original_content="x" * 200,
            output_dir=td,
            file_basename="plotting",
            error_count=10,
            prompt_name="simple",
            model_name="gpt-4o",
            file_type=".md",
        )
        files = os.listdir(td)
    assert "plotting.level_10.gpt-4o_simple.fixed.md" in files
    assert not any(f.endswith(".fixed.Rmd") for f in files)


def test_extension_without_leading_dot_is_normalised(monkeypatch):
    """Accept ``md`` and ``.md`` equivalently — robust against caller typos."""
    _patch_llm(monkeypatch, "---\ntitle: ok\n---\n\nbody\n" * 50)
    with tempfile.TemporaryDirectory() as td:
        bs.fix_with_model(
            llm=None,
            corrupted_content="x" * 200,
            original_content="x" * 200,
            output_dir=td,
            file_basename="plotting",
            error_count=5,
            prompt_name="bioguider",
            model_name="gpt-4o",
            file_type="md",
        )
        files = os.listdir(td)
    assert "plotting.level_5.gpt-4o_bioguider.fixed.md" in files
