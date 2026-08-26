# Phase 3+4b Handoff: Harness Rewrite + Viz Hook + Legacy Delete

**Status**: COMPLETE  
**Owner**: worker-1  
**Task**: #4

## Files Changed

### `system_tests/test_single_file_stress.py`

- **Imports** (top): Added `from langchain_openai import ChatOpenAI`, tenacity retry imports, `from openai import RateLimitError`.
- **MAX_WORKERS**: Changed from hard-coded `16` → `int(os.environ.get("STRESS_MAX_WORKERS", "8"))` (plan Step 16; tunable per proxy health).
- **MODELS dict** (replaced old Azure/Ollama/Claude entries): Now 5 LiteLLM-proxy entries — `gpt-5.4`, `kimi-k2.5`, `glm-5`, `gpt-oss` (model id `gpt-oss-120b` — verify with `curl $OPENAI_BASE_URL/models`), `gpt-4o`. Old keys `gpt4o`/`claude_sonnet`/`qwen3_*/deepseek_*` removed.
- **Deleted `call_ollama` and `call_claude`**: Replaced with `_invoke_with_retry(llm, prompt)` — a tenacity-wrapped helper (`stop=5, wait=exp(2,4,60), retry=RateLimitError`).
- **`fix_with_model` body** collapsed: creates `ChatOpenAI(model=model_id, api_key=$OPENAI_API_KEY, base_url=$OPENAI_BASE_URL)` per call; calls `_invoke_with_retry`. Content cleanup logic preserved unchanged.
- **`save_results` viz hook**: After the markdown report write, added:
  ```python
  try:
      from bioguider.generation.viz import BenchmarkPlotter
      BenchmarkPlotter(output_dir).render_all(output_dir)
  except ImportError:
      print("matplotlib not available; skipping figure generation")
  ```
- **`test_full_benchmark` test_configs**: Replaced 5-tuple list with `[(m, "bioguider") for m in MODELS] + [("gpt-5.4", "simple")]`.
- **`test_all_models_all_levels` test_configs**: Same replacement — 5 bioguider runs + 1 simple baseline (AC6: ≥4 model series).

### `tests/test_viz.py`

- Added `pytest.importorskip("matplotlib", ...)` before the `from bioguider.generation.viz import BenchmarkPlotter` import. Degrades cleanly in envs without matplotlib.

### `bioguider/managers/generation_test_manager.py`

- **Deleted** (orphan — zero callers confirmed via `rg "generation_test_manager\b"`). AC7 ✓

## AC Status After Phase 4

- **AC5**: `save_results` now calls `BenchmarkPlotter.render_all()` which writes `fig{1..6}.{png,pdf}`.
- **AC6**: `test_configs` produces ≥4 model series + 1 simple baseline.
- **AC7**: `rg "generation_test_manager\b"` → zero hits. ✓
- **MAX_WORKERS**: Reduced to 8 default, env-overridable.

## Verification

- Python AST syntax: `test_single_file_stress.py` OK, `test_viz.py` OK.
- AC7 grep: zero hits.
- Live test run gated (Phase 5, team-lead).
