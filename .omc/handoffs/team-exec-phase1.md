# Phase 1 Handoff: LiteLLM Transport Swap

**Status**: COMPLETE  
**Owner**: worker-1  
**Task**: #1

## Changes Made

### `bioguider/agents/agent_utils.py`
- Added `_LITELLM_PROXY_MODEL_SET = {"gpt-4o","gpt-5.4","kimi-k2.5","glm-5","gpt-oss","gpt-oss-120b"}` — widens the old `startswith("gpt")` guard so kimi/glm/gpt-oss no longer raise `ValueError`.
- Added `_TEMP_RESTRICTED_MODELS = {"gpt-5","gpt-5.4","o1","o3"}` — exact-match set replacing the old substring check; `gpt-5.4` no longer silently drops `temperature=0`.
- `get_openai()` now passes `base_url=os.environ.get("OPENAI_BASE_URL")`.
- `get_llm()` gains a `base_url: str = None` parameter. When set, all supported models route through `ChatOpenAI(base_url=...)` and Azure credentials are ignored.
- Empty-string `AZURE_OPENAI_ENDPOINT` normalized to `None` via `azure_endpoint or None` (AC9 / env footgun fix).

### `bioguider/generation/llm_content_generator.py`
- Replaced both inline `get_llm(api_key=os.environ.get(...), ...)` call sites (lines ~114 and ~214) with `get_openai()` — eliminates second call site that hard-read `AZURE_OPENAI_ENDPOINT` directly (AC9).

### `system_tests/conftest.py`
- `get_azure_openai()` renamed to `get_litellm()`, adds `base_url=os.environ.get("OPENAI_BASE_URL", None)` parameter.
- `llm` fixture now calls `get_litellm()`.

### New: `tests/test_litellm_compat.py`
- Parametrized over 5 models: `gpt-5.4`, `kimi-k2.5`, `glm-5`, `gpt-oss-120b`, `gpt-4o`.
- Entire module skipped when `OPENAI_BASE_URL` unset (CI safety via `pytestmark`).
- Asserts `token_usage.total_tokens > 0` per model (AC8 — guards silent zero-accounting on kimi/glm).

### New: `.env.example`
- Documents `OPENAI_BASE_URL`, `OPENAI_API_KEY`, `OPENAI_MODEL` for LiteLLM proxy path.
- Notes Azure vars are ignored when `OPENAI_BASE_URL` is set; embedding vars always required for RAG.

## Third-site caller of `get_llm()`
`rg "get_llm\(" bioguider/` confirms only **one** definition and **one** call site remain inside `bioguider/` (both in `agent_utils.py` itself — definition + `get_openai()` delegation). No third caller found.

## AC Status After Phase 1
- **AC1/AC2** — testable when `OPENAI_BASE_URL` set; `tests/test_litellm_compat.py` covers AC2.
- **AC8** — token-usage assertion in `test_litellm_compat.py`.
- **AC9** — `llm_content_generator.py` no longer hard-reads `AZURE_OPENAI_ENDPOINT`; only `agent_utils.py` and `rag/config.py` (embeddings, intentional) reference it.

## Verification
- Python AST syntax check: all 4 modified/created files OK.
- `poetry install` environment issue (stale lock file, pre-existing) — live test run blocked. Static checks pass.
